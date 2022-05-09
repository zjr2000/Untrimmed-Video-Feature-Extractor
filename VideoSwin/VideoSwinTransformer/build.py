from __future__ import absolute_import

from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint

def build_video_swin_transformer(config, checkpoint):
    """
    config: mmaction config file for model
    checkpoint: path to pre-saved checkpoint file

    return swin video transformer
    """
    cfg = Config.fromfile(config)
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint)
    model = model.backbone
    # print(model)
    return model