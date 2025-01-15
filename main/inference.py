import argparse
import datetime
import inspect
import os, json
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image


import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from einops import rearrange, repeat
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available


from pipelines.pipeline_imagecoductor import ImageConductorPipeline
from modules.unet import UNet3DConditionFlowModel
from utils.visualizer import Visualizer, vis_flow_to_video
from utils.utils import create_image_controlnet, create_flow_controlnet, bivariate_Gaussian, save_videos_grid, load_weights, interpolate_trajectory, load_model
from utils.lora_utils import add_LoRA_to_controlnet

def load_trajectory(traj_path, video_length,resolution=(320,512)): 
    # traj_path不存在文件或文件夹
    if not os.path.exists(traj_path):
        return None
    
    traj = torch.tensor(np.load(traj_path)).float()[:video_length] # [t,h,w,c] -> [c,t,h,w]
    traj[:,:,0]=traj[:,:,0]/resolution[1]
    traj[:,:,1]=traj[:,:,1]/resolution[0]

    traj = torch.clip(traj, min=0.0, max=1.0)
    
    traj[:,:,0]=traj[:,:,0]*resolution[1] -1
    traj[:,:,1]=traj[:,:,1]*resolution[0] -1

    return traj

def view_trainable_param_name(model):
    trainable_name_lists = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            trainable_name_lists.append(name)
    return trainable_name_lists


def points_to_flows(track_points, model_length, height, width):
    input_drag = np.zeros((model_length - 1, height, width, 2))
    for splited_track in track_points:
        if len(splited_track) == 1: # stationary point
            displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
            splited_track = tuple([splited_track[0], displacement_point])
        # interpolate the track
        splited_track = interpolate_trajectory(splited_track, model_length)
        splited_track = splited_track[:model_length]
        if len(splited_track) < model_length:
            splited_track = splited_track + [splited_track[-1]] * (model_length -len(splited_track))
        for i in range(model_length - 1):
            start_point = splited_track[i]
            end_point = splited_track[i+1]
            input_drag[i][int(start_point[1])][int(start_point[0])][0] = end_point[0] - start_point[0]
            input_drag[i][int(start_point[1])][int(start_point[0])][1] = end_point[1] - start_point[1]
    return input_drag

BASIC_MODULE_CACHE = {}
###############################################################################
# 封装一个基础模块加载函数，将 tokenizer, text_encoder, vae 做缓存
###############################################################################
def load_basic_modules(pretrained_model_path, device):
    # key 中包含模型路径以及一个标识（这里使用 "basic_modules"）
    key = (pretrained_model_path, "basic_modules")
    if key in BASIC_MODULE_CACHE:
        tokenizer, text_encoder, vae = BASIC_MODULE_CACHE[key]
        print("[INFO] Loading Tokenizer/TextEncoder/VAE because cache hit ...")
    else:
        print("[INFO] Loading Tokenizer/TextEncoder/VAE because cache miss ...")
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
        BASIC_MODULE_CACHE[key] = (tokenizer, text_encoder, vae)
    return tokenizer, text_encoder, vae

MODEL_CACHE={}
###############################################################################
# 封装一个 UNet / ControlNet 的加载函数，避免重复加载
###############################################################################
def load_unet_controlnets(model_config, args, inference_config):
    # 将可能影响模型加载的关键路径或超参，组合成一个 key，用于判断缓存
    key = (
        model_config.get("unet_path", ""),
        model_config.get("image_controlnet_path", ""),
        model_config.get("flow_controlnet_path", ""),
        args.pretrained_model_path,
        str(inference_config.unet_additional_kwargs),
    )
    if key in MODEL_CACHE:
        unet, image_controlnet, flow_controlnet = MODEL_CACHE[key]
        print("[INFO] Loading UNet/ControlNets because cache hit ...")
    else:
        print("[INFO] Loading UNet/ControlNets because cache miss ...")

        # 初始化 UNet
        unet = UNet3DConditionFlowModel.from_pretrained_2d(
            args.pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
        )

        # 初始化 image_controlnet
        image_controlnet = None
        if model_config.image_controlnet_config is not None:
            image_controlnet = create_image_controlnet(model_config.image_controlnet_config, unet)

        # 初始化 flow_controlnet
        flow_controlnet = None
        if model_config.flow_controlnet_config is not None:
            flow_controlnet = create_flow_controlnet(model_config.flow_controlnet_config, unet)
            add_LoRA_to_controlnet(args.lora_rank, flow_controlnet)

        # 加载预训练权重
        unet_path = model_config.get("unet_path", "")
        image_controlnet_path = model_config.get("image_controlnet_path", "")
        flow_controlnet_path  = model_config.get("flow_controlnet_path", "")

        load_model(unet, unet_path)
        load_model(image_controlnet, image_controlnet_path)
        load_model(flow_controlnet, flow_controlnet_path)

        # 更新缓存
        MODEL_CACHE[key] = (unet, image_controlnet, flow_controlnet)

    return unet, image_controlnet, flow_controlnet

@torch.no_grad()
def main(args):

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/" # {time_str}-{args.video_info}
    os.makedirs(savedir,exist_ok=True)

    config  = OmegaConf.load(args.config)
    samples = []
    lora_rank = args.lora_rank

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    blur_kernel = bivariate_Gaussian(kernel_size=99, sig_x=10, sig_y=10, theta=0, grid=None, isotropic=True)

    # create validation pipeline
    # tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    # vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

     # 先加载一次基础模块 (tokenizer, text_encoder, vae)
    tokenizer, text_encoder, vae = load_basic_modules(args.pretrained_model_path, device)


    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        
        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

        # 从缓存或新加载 UNet / ControlNet
        unet, image_controlnet, flow_controlnet = load_unet_controlnets(model_config, args, inference_config)

        # unet = UNet3DConditionFlowModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

        # ### >>> Initialize image controlnet >>> ###
        # if model_config.image_controlnet_config is not None:
        #     image_controlnet = create_image_controlnet(model_config.image_controlnet_config, unet)
        
        # ### >>> Initialize flow controlnet >>> ###
        # if model_config.flow_controlnet_config is not None:
        #     flow_controlnet = create_flow_controlnet(model_config.flow_controlnet_config, unet)
        #     add_LoRA_to_controlnet(lora_rank, flow_controlnet)

        
        # # Load pretrained unet weights
        # unet_path = model_config.get("unet_path", "")
        # load_model(unet, unet_path)
        
        # # Load pretrained image controlnet weights
        # image_controlnet_path = model_config.get("image_controlnet_path", "")
        # load_model(image_controlnet, image_controlnet_path)

        # # Load pretrained flow controlnet weights
        # flow_controlnet_path = model_config.get("flow_controlnet_path", "")
        # load_model(flow_controlnet, flow_controlnet_path)

        # load image condition
        controlnet_images = None
        if model_config.get("controlnet_images", "") != "":
            assert model_config.get("controlnet_images", "") != ""            
        
            image_paths = model_config.controlnet_images
            image_paths = args.image_path # 改用外部传入的图片
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                # transforms.RandomResizedCrop(
                #     (model_config.H, model_config.W), (1.0, 1.0), 
                #     ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                # ),
                transforms.Resize((model_config.H, model_config.W)),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            # 评测生成不需要，去掉
            # os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            # for i, image in enumerate(controlnet_images):
            #     Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/sample{model_idx}_{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")
            
            num_controlnet_images = controlnet_images.shape[2]
            controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
            controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
            controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # load traj condition
        controlnet_flows = None
        if model_config.get("controlnet_trajs", "") != "":
            # import ipdb; ipdb.set_trace()

            track_ponints_path= model_config.controlnet_trajs
            # with open(track_ponints_path, 'r') as f:
            #     track_ponints = json.load(f)
            track_ponints = load_trajectory(args.traj_path,args.L).permute(1,0,2) #.permute(1,0,2) # .tolist() # 改用外部传入的轨迹
            print(track_ponints.shape) 
 
            controlnet_flows = points_to_flows(track_ponints, model_config.L, model_config.H, model_config.W) #  (15, 256, 384, 2)
            print(f"controlnet_flows shape : {controlnet_flows.shape}")  

            
            # controlnet_flows = args.traj.permute(1,2,3,0).detach().cpu().numpy()  # 改用外部传入的光流 controlnet_flows shape : torch.Size([2, 16, 320, 512]) -> [16,320,512,2]
            # print(f"our controlnet_flows shape : {controlnet_flows.shape}")  

          
            for i in range(0, model_config.L-1):
                controlnet_flows[i] = cv2.filter2D(controlnet_flows[i], -1, blur_kernel)
            
            controlnet_flows = np.concatenate([np.zeros_like(controlnet_flows[0])[np.newaxis, ...], controlnet_flows], axis=0)  # pad the first frame with zero flow
            # os.makedirs(os.path.join(savedir, "control_flows"), exist_ok=True)
            trajs_video = vis_flow_to_video(controlnet_flows, num_frames=model_config.L) # T-1 x H x W x 3
            # 光流图保存
            # 评测生成不需要，去掉
            # torchvision.io.write_video(f'{savedir}/control_flows/sample{model_idx}_train_flow.mp4', trajs_video, fps=8, video_codec='h264', options={'crf': '10'})

            controlnet_flows = torch.from_numpy(controlnet_flows)[None].to(controlnet_images)[:, :model_config.L, ...]
            controlnet_flows =  rearrange(controlnet_flows, "b f h w c-> b c f h w")


        unet.to(device)
        image_controlnet.to(device)
        flow_controlnet.to(device)
        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()

        pipeline = ImageConductorPipeline(
            unet=unet,
            vae=vae, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            image_controlnet=image_controlnet,
            flow_controlnet=flow_controlnet,
        ).to(device)

        
        # load motion_module & domain adapter & dreambooth_model (optional), see Animatediff
        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to(device)
      
        
        # prompts      = model_config.prompt
        prompts      = [args.caption]  # 改用外部传入的caption
        print(prompts)
        n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        print("n_prompt",n_prompts)
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        
        config[model_idx].random_seed = []
        
        # traj_guidance_scales = [x / 2 for x in range(2, 7)]
        # traj_guidance_scales = [1.1]
        traj_guidance_scales = [1]
        vis = Visualizer(save_dir=f"{savedir}/sample", pad_value=0, linewidth=2, mode='cool', tracks_leave_trace=-1)


        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            
            # manually set random seed for reproduction
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())
            
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {args.video_info} ...")
            assert model_config.control_mode in ["object", "camera"], "control_mode in [object, camera]"
            sample = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,
                controlnet_images = controlnet_images,
                controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                controlnet_flows  = controlnet_flows,
                control_mode = model_config.control_mode,
                eval_mode = True,
            ).videos
            samples.append(sample)
            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            # 最终生成的视频
            # save_videos_grid(sample, f"{savedir}/sample/{args.video_info}.mp4") 
            save_videos_grid(sample, f"{savedir}/{args.video_info}.mp4") 

            if model_config.control_mode == "object":
                track_ponints_new = []
                for point in track_ponints:
                    # print(point.shape)
                    splited_track = interpolate_trajectory(point, 16)
                    track_ponints_new.append(splited_track)
                points_track_vis = np.array(track_ponints_new).transpose(1, 0, 2)
                vis_video_obj= (sample[0] * 255).numpy().astype(np.uint8).transpose(1, 0, 2, 3)
                # 最终生成的视频+轨迹可视化图，其实可以省去
                # 评测生成不需要，去掉
                # vis.visualize(torch.from_numpy(vis_video_obj[None]), torch.from_numpy(points_track_vis[None]), filename=f"track-traj-{sample_idx}-{args.video_info}", query_frame=0)

            print(f"save to {savedir}/sample/{args.video_info}.mp4") # # 最终视频
            sample_idx += 1

        # samples = torch.concat(samples)
        # 这个应该是batch拼起来的最终视频结果
        # 评测生成不需要，去掉
        # save_videos_grid(samples, f"{savedir}/{model_idx}-sample.gif", n_rows=4) # 最终视频
        samples = []

    # 评测生成不需要，去掉
    # OmegaConf.save(config, f"{savedir}/config.yaml")


def run_imageconductor_inference(
        image_path=None,
        traj_path=None,
        traj=None,
        caption=None,
        video_info=None,
        config=None,
        pretrained_model_path="../../model/stable-diffusion-v1-5",
        inference_config="configs/inference/inference.yaml", 
        L=16, 
        W=384,# 512,#384,
        H=256,#320,#256,
        lora_rank=64, 
        without_xformers=False
        ):
    
    # 创建一个命名空间对象，模拟argparse的args
    class Args:
        pass
    
    args = Args()
    args.image_path = image_path
    args.traj_path = traj_path
    args.traj = traj
    args.caption = caption
    args.video_info = video_info
    args.pretrained_model_path = pretrained_model_path
    args.inference_config = inference_config
    args.config = config
    args.L = L
    args.W = W
    args.H = H
    args.lora_rank = lora_rank
    args.without_xformers = without_xformers
    print(f"args.image_path: {args.image_path}")
    print(f"args.traj_path: {args.traj_path}")
    print(f"args.traj.shape: {traj.shape}")
    print(f"args.caption: {args.caption}")
    print(f"args.video_info: {args.video_info}")
    
    main(args)


if __name__ == "__main__":
    run_imageconductor_inference(config="configs/prompt/trajs/object_evaluate.yaml")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained-model-path", type=str, default="../../model/stable-diffusion-v1-5",)
    # parser.add_argument("--inference-config",      type=str, default="configs/inference/inference.yaml")    
    # parser.add_argument("--config",                type=str, required=True)
    
    # parser.add_argument("--L", type=int, default=16 )
    # parser.add_argument("--W", type=int, default=384)
    # parser.add_argument("--H", type=int, default=256)
    # parser.add_argument("--lora_rank", type=int, default=64)
    # parser.add_argument("--without-xformers", action="store_true")

    # args = parser.parse_args()
    # main(args)
