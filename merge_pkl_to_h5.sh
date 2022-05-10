FEATURE_FOLDER=anet_swin_feat/fps_15_len_16_stride_16
SAVE_PATH=anet_swin_feat/anet_swin_fps_15_len_16_stride_16.h5

python TSP/extract_features/merge_pkl_files_into_one_h5_feature_file.py \
--features-folder $FEATURE_FOLDER \
--output-h5 $SAVE_PATH