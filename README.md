# Untrimmed Video Feature Extractor
Long and untrimmed video learning has recieved increasing attention in recent years. This repo aims to provide some simple and effective scripts for long and untrimmed video feature extraction. We adopt the video processing pipline from [TSP](https://github.com/HumamAlwassel/TSP) and adapt it with several awesome vision pretraining backbones.

## Environment
Run `conda env create -f environment.yml` for base environment setup. For specific model setup, please check their project link:

[Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)

[Omnivore](https://github.com/facebookresearch/omnivore)

[CLIP](https://github.com/openai/CLIP)

## Usage
Run ```bash Scripts/generate_video_metada.sh``` to extract metadata from video, where ```VIDEO_FOLDER``` is the directory contains the raw videos, and ```OUTPUT_CSV_PATH``` is the output csv file contains the generated video metadata.

Then run the following script to extract features:
```sh
bash Scripts/extract_${MODEL_NAME}_feat.sh
```
Before running, rember to set the defined variable in the script.

Finally, run ```bash Scripts/merge_pkl_to_h5.sh``` to merge the video features to a single h5 file.

## Acknowledgement
This repo is mainly based on pipeline provided by [TSP](https://github.com/HumamAlwassel/TSP).
