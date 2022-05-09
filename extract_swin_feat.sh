#!/bin/bash -i

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

DATA_PATH=/devdata1/VideoCaption/ActivityNet/processd_raw_videos/all/
METADATA_CSV_FILENAME=/devdata1/VideoCaption/FeatureExtraction/activitynet_v1-3_test_metadata.csv

CONFIG_PATH=VideoSwin/VideoSwinTransformer/configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py
CHECKPOINT_PATH=VideoSwin/VideoSwinTransformer/swin_checkpoints/swin_base_patch244_window877_kinetics600_22k.pth

# Choose the stride between clips, e.g. 16 for non-overlapping clips and 1 for dense overlapping clips
CLIP_LEN=16
STRIDE=16
FRAME_RATE=15

# Optional: Split the videos into multiple shards for parallel feature extraction
# Increase the number of shards and run this script independently on separate GPU devices,
# each with a different SHARD_ID from 0 to NUM_SHARDS-1.
# Each shard will process (num_videos / NUM_SHARDS) videos.
SHARD_ID=0
NUM_SHARDS=1
DEVICE=cuda

if [ -z "$DATA_PATH" ]; then
    echo "DATA_PATH variable is not set."
    echo "Please set DATA_PATH to the folder containing the videos you want to process."
    exit 1
fi

if [ -z "$METADATA_CSV_FILENAME" ]; then
    echo "METADATA_CSV_FILENAME variable is not set."
    echo "We provide metadata CSV files for ActivityNet and THUMOS14 in the data folder."
    exit 1
fi

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "CHECKPOINT_PATH variable is not set."
    echo "Please set CHECKPOINT_PATH to the location of the local checkpoint .pth file."
    echo "Make sure to set the correct BACKBONE variable as well."
    exit 1
fi

if [ -z "$CONFIG_PATH" ]; then
    echo "CONFIG_PATH variable is not set."
    exit 1
fi

####################################################################################
############################# PARAMETERS TO KEEP AS IS #############################
####################################################################################

OUTPUT_DIR=anet_swin_feat/fps_${FRAME_RATE}_len_${CLIP_LEN}_stride_${STRIDE}/

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH="$(dirname $0)/VideoSwin/VideoSwinTransformer":$PYTHONPATH \
python VideoSwin/extract_swin_feat_fast.py \
--data-path $DATA_PATH \
--metadata-csv-filename $METADATA_CSV_FILENAME \
--config_path $CONFIG_PATH \
--checkpoint_path $CHECKPOINT_PATH \
--frame-rate $FRAME_RATE \
--clip-len $CLIP_LEN \
--stride $STRIDE \
--shard-id $SHARD_ID \
--num-shards $NUM_SHARDS \
--device $DEVICE \
--output-dir $OUTPUT_DIR