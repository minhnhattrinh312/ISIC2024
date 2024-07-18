# Path: segment2d/config.py
from yacs.config import CfgNode as CN

cfg = CN()
cfg.DATA = CN()
cfg.TRAIN = CN()
cfg.SYS = CN()
cfg.OPT = CN()
cfg.DIRS = CN()
cfg.PREDICT = CN()
cfg.TRAIN.CONVEXT = CN()

cfg.DATA.NUM_CLASS = 2
cfg.DATA.CLASS_WEIGHT = [0.05, 0.95]  # default [0.1, 0.9]
cfg.DATA.IN_CHANNEL = 3

cfg.DATA.INDIM_MODEL = 2
cfg.TRAIN.PRETRAIN = True
cfg.TRAIN.PRETRAIN = True

cfg.TRAIN.CONVEXT.IN22K = True
cfg.TRAIN.CONVEXT.DROPOUT = 0.2
