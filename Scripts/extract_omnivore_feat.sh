#!/bin/bash -i

####################################################################################
########################## PARAMETERS THAT NEED TO BE SET ##########################
####################################################################################

DATA_PATH=
METADATA_CSV_FILENAME=
MODEL_NAME=omnivore_swinL_imagenet21k
OUTPUT_DIR=
# Choose the stride between clips, e.g. STRIDE = CLIP_LEN for non-overlapping clips and STRIDE = 1 for dense overlapping clips
CLIP_LEN=
STRIDE=
FRAME_RATE=

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

if [ -z "$OUTPUT_DIR" ]; then
    echo "OUTPUT_DIR is not set."
    exit 1
fi

####################################################################################
############################# PARAMETERS TO KEEP AS IS #############################
####################################################################################
# OUTPUT_DIR=anet_omnivore_feat/fps_${FRAME_RATE}_len_${CLIP_LEN}_stride_${STRIDE}/

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=`pwd`:$PYTHONPATH \
python omnivore/extract_omnivore_feature.py \
--data-path $DATA_PATH \
--metadata-csv-filename $METADATA_CSV_FILENAME \
--model-name $MODEL_NAME \
--frame-rate $FRAME_RATE \
--clip-len $CLIP_LEN \
--stride $STRIDE \
--shard-id $SHARD_ID \
--num-shards $NUM_SHARDS \
--device $DEVICE \
--output-dir $OUTPUT_DIR
