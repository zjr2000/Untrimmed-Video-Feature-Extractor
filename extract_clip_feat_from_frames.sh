#!/bin/bash -i

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

DATA_PATH=

CLIP_BACKBONE=ViT-L/14@336px
# Optional: Split the videos into multiple shards for parallel feature extraction
# Increase the number of shards and run this script independently on separate GPU devices,
# each with a different SHARD_ID from 0 to NUM_SHARDS-1.
# Each shard will process (num_videos / NUM_SHARDS) videos.
DEVICE=cuda

if [ -z "$DATA_PATH" ]; then
    echo "DATA_PATH variable is not set."
    echo "Please set DATA_PATH to the folder containing the videos you want to process."
    exit 1
fi


if [ -z "$CLIP_BACKBONE" ]; then
    echo "CLIP_BACKBONE variable is not set."
    exit 1
fi

####################################################################################
############################# PARAMETERS TO KEEP AS IS #############################
####################################################################################

OUTPUT_DIR=youmakeup_clip_feat

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 \
python CLIP/extract_clip_feature_from_frames.py \
--data-path $DATA_PATH \
--clip_backbone $CLIP_BACKBONE \
--device $DEVICE \
--output-dir $OUTPUT_DIR
