import os
import json
import torch
import numpy as np
from PIL import Image
import pandas as pd

from main.inference import run_imageconductor_inference

# 假设这些变量已经定义


metadata = pd.read_csv("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/webvid10m/webvid10m/data/val/partitions/0000.csv")

def load_webvid(video_length=16):
    # load json /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/webvid10m/.123/.asdf/.09opi/ViewCrafter/dataset_pipeline/successful_subdirs_val_1000.json
    with open('/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/webvid10m/.123/.asdf/.09opi/ViewCrafter/dataset_pipeline/successful_subdirs_val_1000.json', 'r') as f:
        img_dirs = json.load(f)
    
    # 替换img_dir = img_dirs[index].replace(old_path, new_path)
    old_path = '/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhengsixiao-240207140179/datasets/webvid10m'
    new_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/zhengsixiao/webvid10m/webvid10m'
    img_dirs = [img_dir.replace(old_path, new_path) for img_dir in img_dirs]
 
    # 提取page_dir, videoid, start_frame, frame_stride

    all_traj_path = []
    all_frame_paths = []
    all_caption = []

    results = []
    
    for img_dir in img_dirs:
        page_dir, videoid, start_frame, frame_stride,video_info = extract_info(img_dir)

        # 获取caption
        caption = retrieve_metadata(page_dir, videoid)
    
        # 获取frame_paths
        frame_paths = get_frame_paths(img_dir,video_length) # 采样video_length帧
    
        # 获取traj_path
        traj_path_coord = os.path.join(os.path.dirname(img_dir), f"sparse_trajs_coord1.npy") # 位置序列
        traj_path_optical = os.path.join(os.path.dirname(img_dir), f"sparse_gs1.npy") # 稀疏光流
        traj_path_optical = os.path.join(os.path.dirname(img_dir), f"dense_gs.npy") # 试一试稠密光流

        result = {
            'frame_paths': frame_paths,
            'traj_paths_coord': traj_path_coord,
            'traj_paths_optical': traj_path_optical,
            'captions': caption,
            'video_info': video_info,
        }
        results.append(result)
    
    return results

def extract_info(img_dir):
    parts = img_dir.split('/')
    page_dir = parts[-3]
    video_info = parts[-2]
    videoid, start_frame, frame_stride = video_info.split('_')
    return page_dir, videoid, int(start_frame), int(frame_stride),video_info

def retrieve_metadata(page_dir, videoid):
    # 加载含有caption的csv
    df = metadata
    # 获取与page_dir, videoid一致对应的caption
    caption = df[(df['page_dir'] == page_dir) & (df['videoid'] == int(videoid))]['name'].values[0]
    print(f"video={videoid} retrieve Caption: {caption}")
    
    return caption

def get_frame_paths(img_dir,video_length):
    frame_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])[:video_length]
    return frame_paths
 
def load_trajectory(traj_path, video_length,resolution=(320,512),traj_type="coord"): # 原始的resolution=(256,384)
    # traj_path不存在文件或文件夹
    if not os.path.exists(traj_path):
        return None
    
    # 位置序列
    if traj_type=="coord":
        traj = torch.tensor(np.load(traj_path)).float()[:video_length] # [t,h,w,c] -> [c,t,h,w]
        traj[:,:,0]=traj[:,:,0]/resolution[1]
        traj[:,:,1]=traj[:,:,1]/resolution[0]
    
        padded_traj = torch.zeros(video_length, 256, 2)
        padded_traj[:, :traj.size(1), :] = traj
        padded_traj = torch.clip(padded_traj, min=0.0, max=1.0)
    # 光流
    else: 
        traj = torch.tensor(np.load(traj_path)).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        # resize
        from torchvision.transforms import Resize
        traj =  Resize(resolution, interpolation=Image.BICUBIC)(traj)
        traj =  traj[:,:video_length]
        
    # return padded_traj
    return traj


def load_images(frame_paths):
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as img:
            img = img.convert('RGB')
            img_array = np.array(img).astype(np.float32)
            frames.append(img_array)
    frames = np.stack(frames)
    frames = torch.tensor(frames).permute(3, 0, 1, 2).float()
    # 进行空间变换，如果需要
    # frames = spatial_transform(frames)
    # 转换到[-1,1]
    frames = (frames / 255 - 0.5) * 2
    return frames


def run(webvid10_results,traj_type="coord"):

    for idx,item in enumerate(webvid10_results):
        print(f"Run inference for [{idx}]")
        frame_paths = item['frame_paths']
        # traj_path = item['traj_paths']
        traj_path_coord = item['traj_paths_coord']
        traj_path_optical = item['traj_paths_optical']
        caption = item['captions']
        video_info = item['video_info']

        # 加载轨迹
        if traj_type=="coord":
            traj_path = traj_path_coord
        elif traj_type=="optical_flow":
            traj_path = traj_path_optical
        
        # resolution=(256,384)
        resolution=(320,512) # (256,384)
        traj = load_trajectory(traj_path, video_length,resolution=resolution,traj_type=traj_type)
        if traj==None:
            continue


        print(f"video_info=={video_info}")
        print(f"frame_paths: {frame_paths[0]}...")
        print(f"traj_path: {traj_path}")
        print(f"caption: {caption}")
        print(f"webvid10 traj Load shape: {traj.shape}")

        # 跑起来
        config="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/pengzimian-241108540199/project/ImageConductor/configs/prompt/trajs/object_evaluate.yaml"
        run_imageconductor_inference(frame_paths[0],traj_path,traj,caption,video_info,config)

    print(cnt)
    
    # 进行处理
    # ...
    print("运行处理函数，数据已加载。")
    # 这里可以添加数据处理的代码
    
if __name__ == "__main__":
    video_length=16
    webvid10_results = load_webvid(video_length)
    print(f"test num: {len(webvid10_results)}")
    run(webvid10_results)