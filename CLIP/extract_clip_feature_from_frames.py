import clip
import torch
import os
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle as pkl


class RawFramesDataset(Dataset):
    def __init__(self, root_folder, preprocess, output_dir):
        self.all_frame_paths = self.load_all_frame_paths(root_folder)
        self.preprocess = preprocess
        self.output_dir = output_dir
        self.saved_features = {}

    def load_all_frame_paths(self, root_folder):
        all_frame_paths = []
        video_names = os.listdir(root_folder)
        for video_name in video_names:
            video_folder = os.path.join(root_folder, video_name)
            frame_paths = os.listdir(video_folder)
            frame_paths = sorted(frame_paths)
            for idx, frame_path in enumerate(frame_paths):
                is_last = idx == len(frame_paths)-1
                all_frame_paths.append((video_name, os.path.join(video_folder, frame_path), is_last))
        return all_frame_paths
            
    def __len__(self):
        return len(self.all_frame_paths)

    def __getitem__(self, index):
        video_name, image_path, is_last = self.all_frame_paths[index]
        frame = self.preprocess(Image.open(image_path))
        return {'filename':video_name, 'clip':frame, 'is-last-clip':is_last}

    def save_features(self, batch_features, batch_input):
        batch_features = batch_features.detach().cpu().numpy()

        for i in range(batch_features.shape[0]):
            filename, is_last_clip = batch_input['filename'][i], batch_input['is-last-clip'][i]
            if not (filename in self.saved_features):
                self.saved_features[filename] = []
            self.saved_features[filename].append(batch_features[i,...])

            if is_last_clip:
                # dump features to disk at self.output_dir and remove them from self.saved_features
                output_filename = os.path.join(self.output_dir, filename + '.pkl')
                self.saved_features[filename] = np.stack(self.saved_features[filename])
                with open(output_filename, 'wb') as fobj:
                    pkl.dump(self.saved_features[filename], fobj)
                del self.saved_features[filename]


def main(args):
    print(args)
    print('TORCH VERSION: ', torch.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model, transform = clip.load(args.clip_backbone, device=device)

    dataset = RawFramesDataset(
        root_folder=args.data_path,
        preprocess=transform,
        output_dir=args.output_dir
        )

    print('CREATING DATA LOADER')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    save_folder = args.output_dir
    print('START FEATURE EXTRACTION')

    model.eval()
    with torch.no_grad():
        for sample in tqdm(data_loader):
            frames = sample['clip'].to(device, non_blocking=True)
            feat = model.encode_image(frames)
            data_loader.dataset.save_features(feat, sample)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Features extraction script')

    parser.add_argument('--data-path', required=True,
                        help='Path to the directory containing the videos files')

    parser.add_argument('--clip_backbone', required=True,
                        help='Clip backbone')

    parser.add_argument('--device', default='cuda',
                        help='Device to train on (default: cuda)')

    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loading workers (default: 6)')

    parser.add_argument('--output_dir', required=True,
                        help='Path for saving features')

    args = parser.parse_args()
    main(args)