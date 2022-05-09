import re
import torch
from torchvision.io import read_video
from torchvision.transforms import transforms
import mmcv
import numpy as np
from TSP.extract_features.eval_video_dataset import EvalVideoDataset
import os
import torchvision
import pandas as pd
import sys
import torch.nn.functional as F
from VideoSwinTransformer.build import build_video_swin_transformer
from einops import rearrange
import argparse
from tqdm import tqdm

class ResizeAndToTensor(object):
    def __call__(self, frames):
        to_tensor = transforms.ToTensor()
        sampled_frames = []
        frames = frames.numpy().astype('float32')
        for frame in frames:
            img_h, img_w = frame.shape[0], frame.shape[1]
            scale = (np.inf, 256)
            new_w, new_h = mmcv.rescale_size((img_w, img_h), scale)
            frame = mmcv.imresize(frame, (new_w, new_h))
            sampled_frames.append(frame)
        sampled_frames = np.stack(sampled_frames)
        frames = torch.cat([to_tensor(f).unsqueeze(0) for f in sampled_frames])
        return frames

class Rearrange(object):
    def __call__(self, data):
        data = rearrange(data, 'T C H W -> C T H W')
        return data


def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for sample in tqdm(data_loader):
            clip = sample['clip'].to(device, non_blocking=True)
            feat = model(clip)
            feat = F.adaptive_avg_pool3d(feat, (1,1,1))
            feat = feat.view(feat.shape[0], -1)
            data_loader.dataset.save_features(feat, sample)


def main(args):
    print(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print('LOADING DATA')
    transform = transforms.Compose([
        ResizeAndToTensor(),
        transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        transforms.CenterCrop((224, 224)),
        Rearrange()
    ])

    metadata_df = pd.read_csv(args.metadata_csv_filename)
    shards = np.linspace(0,len(metadata_df),args.num_shards+1).astype(int)
    start_idx, end_idx = shards[args.shard_id], shards[args.shard_id+1]
    print(f'shard-id: {args.shard_id + 1} out of {args.num_shards}, '
        f'total number of videos: {len(metadata_df)}, shard size {end_idx-start_idx} videos')

    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()
    metadata_df['is-computed-already'] = metadata_df['filename'].map(lambda f:
        os.path.exists(os.path.join(args.output_dir, os.path.basename(f).split('.')[0] + '.pkl')))
    metadata_df = metadata_df[metadata_df['is-computed-already']==False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

    dataset = EvalVideoDataset(
        metadata_df=metadata_df,
        root_dir=args.data_path,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        stride=args.stride,
        output_dir=args.output_dir,
        transforms=transform)

    print('CREATING DATA LOADER')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    

    print(f'LOADING MODEL')
    model = build_video_swin_transformer(args.config_path, args.checkpoint_path)
    model.to(device)

    print('START FEATURE EXTRACTION')
    evaluate(model, data_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Features extraction script')

    parser.add_argument('--data-path', required=True,
                        help='Path to the directory containing the videos files')
    parser.add_argument('--metadata-csv-filename', required=True,
                        help='Path to the metadata CSV file')

    parser.add_argument('--config_path', required=True,
                        help='Path to mmaction model config file')
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to Video Swin Transformer checkpoint')

    parser.add_argument('--clip-len', default=16, type=int,
                        help='Number of frames per clip (default: 16)')
    parser.add_argument('--frame-rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled (default: 15)')
    parser.add_argument('--stride', default=16, type=int,
                        help='Number of frames (after resampling with frame-rate) between consecutive clips (default: 16)')

    parser.add_argument('--device', default='cuda',
                        help='Device to train on (default: cuda)')

    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size per GPU (default: 32)')
    parser.add_argument('--workers', default=6, type=int,
                        help='Number of data loading workers (default: 6)')

    parser.add_argument('--output-dir', required=True,
                        help='Path for saving features')
    parser.add_argument('--shard-id', default=0, type=int,
                        help='Shard id number. Must be between [0, num-shards)')
    parser.add_argument('--num-shards', default=1, type=int,
                        help='Number of shards to split the metadata-csv-filename')

    args = parser.parse_args()
    main(args)
