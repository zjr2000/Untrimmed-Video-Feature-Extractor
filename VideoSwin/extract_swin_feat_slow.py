import sys
import mmcv
import os
import numpy as np
sys.path.append('/devdata1/VideoCaption/SLECap/models/VideoSwinTransformer')
import torch
from build import build_video_swin_transformer
from torchvision import transforms
from einops import rearrange
import h5py
import torch.nn.functional as F
from tqdm import tqdm

def process_frames(frames):
    to_tensor = transforms.ToTensor()
    mean=[123.675, 116.28, 103.53] 
    std=[58.395, 57.12, 57.375]
    compose = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])
    sampled_frames = []
    for frame in frames:
        frame = mmcv.bgr2rgb(frame)
        img_h, img_w = frame.shape[0], frame.shape[1]
        scale = (np.inf, 256)
        new_w, new_h = mmcv.rescale_size((img_w, img_h), scale)
        frame = mmcv.imresize(frame, (new_w, new_h))
        sampled_frames.append(frame)
    sampled_frames = np.stack(sampled_frames)
    frames = torch.cat([to_tensor(f).unsqueeze(0) for f in sampled_frames])
    frames = compose(frames)
    return frames

if __name__ == '__main__':
    config_path = '/devdata1/VideoCaption/SLECap/models/VideoSwinTransformer/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
    checkpoint_path = '/devdata1/VideoCaption/SLECap/models/VideoSwinTransformer/swin_checkpoints/swin_base_patch244_window877_kinetics600_22k.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_video_swin_transformer(config_path, checkpoint_path)
    model.to(device)
    video_dir = '/devdata1/VideoCaption/ActivityNet/processd_raw_videos/all/'
    save_path = '/devdata1/VideoCaption/ActivityNet/vid_swin_feat_1024_interval_8.h5'

    h5_file = h5py.File(save_path, 'w')
    interval = 8
    batch_size = 16
    for video_name in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_name)
        frames = mmcv.VideoReader(video_path)
        print('frame num:', len(frames))
        frames = process_frames(frames)
        clips = []
        size = len(frames) - len(frames) % 8
        for center_idx in range(interval, size, interval):
            current_clip = frames[center_idx - interval: center_idx + interval,]
            current_clip = current_clip.unsqueeze(0)
            clips.append(current_clip)
        clips = torch.cat(clips, dim=0)
        clips = rearrange(clips, 'B T C H W -> B C T H W')
        batches = []
        for i in range(0, len(clips), batch_size):
            batches.append(clips[i:i+batch_size])
            
        # (D T/2 H/32 W/32)
        clip_feat = []
        with torch.no_grad():
            for data in batches:
                data = data.to(device)
                feat = model(data)
                feat = F.adaptive_avg_pool3d(feat, (1,1,1))
                feat = feat.view(feat.shape[0], -1)
                clip_feat.append(feat)
        clip_feat = torch.cat(clip_feat, dim=0)
        clip_feat = clip_feat.cpu().numpy()
        print(video_name, clip_feat.shape)
        h5_file.create_dataset(video_name, data=clip_feat)
