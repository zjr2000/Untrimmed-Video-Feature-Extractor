import torch
from typing import Any
from torchvision.transforms import transforms as T
import numpy as np
import os
import sys
from TSP.extract_features.eval_video_dataset import EvalVideoDataset
import torchvision
import pandas as pd
import torch.nn.functional as F
from einops import rearrange
import argparse
from tqdm import tqdm
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms._transforms_video import NormalizeVideo
import models.omnivore_model as omni_model

class Rearrange():
    def __call__(self, data):
        data = rearrange(data, 'T H W C -> C T H W')
        return data

# T*H*W*C
video_transform = transform=T.Compose(
        [
            Rearrange(),
            T.Lambda(lambda x: x / 255.0),  
            ShortSideScale(size=224),
            T.CenterCrop((224, 224)),
            NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
        
def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for sample in tqdm(data_loader):
            clip = sample['clip'].to(device, non_blocking=True)
            feat = model(clip)
            feat = feat[0]
            data_loader.dataset.save_features(feat, sample)


def main(args):
    print(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print('LOADING DATA')

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
        transforms=video_transform)

    print('CREATING DATA LOADER')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    

    print(f'LOADING MODEL')

    # Model input shape: B*C*T*H*W
    if args.model_name == "omnivore_swinT":
        model = omni_model.omnivore_swinT(load_heads=False)
    if args.model_name == "omnivore_swinS":
        model = omni_model.omnivore_swinS(load_heads=False)
    if args.model_name == "omnivore_swinB":
        model = omni_model.omnivore_swinB(load_heads=False)
    if args.model_name == "omnivore_swinB_imagenet21k":
        model = omni_model.omnivore_swinB_imagenet21k(load_heads=False)
    if args.model_name == "omnivore_swinL_imagenet21k":
        model = omni_model.omnivore_swinL_imagenet21k(load_heads=False)
    if args.model_name == "omnivore_swinL_imagenet21k":
        model = omni_model.omnivore_swinL_imagenet21k(load_heads=False)
    if args.model_name == "omnivore_swinB_epic":
        model = omni_model.omnivore_swinB_epic(load_heads=False)
    model.to(device)

    print('START FEATURE EXTRACTION')
    evaluate(model, data_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Features extraction script')

    parser.add_argument('--data-path', required=True,
                        help='Path to the directory containing the videos files')
    parser.add_argument('--metadata-csv-filename', required=True,
                        help='Path to the metadata CSV file')
    parser.add_argument("--model-name", default="omnivore_swinB",
                        help='Name of the model')

    parser.add_argument('--clip-len', default=16, type=int,
                        help='Number of frames per clip (default: 16)')
    parser.add_argument('--frame-rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled (default: 15)')
    parser.add_argument('--stride', default=16, type=int,
                        help='Number of frames (after resampling with frame-rate) between consecutive clips (default: 16)')

    parser.add_argument('--device', default='cuda',
                        help='Device to train on (default: cuda)')

    parser.add_argument('--batch-size', default=45, type=int,
                        help='Batch size per GPU (default: 10)')
    parser.add_argument('--workers', default=45, type=int,
                        help='Number of data loading workers (default: 3)')

    parser.add_argument('--output-dir', required=True,
                        help='Path for saving features')
    parser.add_argument('--shard-id', default=0, type=int,
                        help='Shard id number. Must be between [0, num-shards)')
    parser.add_argument('--num-shards', default=1, type=int,
                        help='Number of shards to split the metadata-csv-filename')

    args = parser.parse_args()
    main(args)
