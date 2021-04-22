from easydict import EasyDict

Cfg = EasyDict()
Cfg.augment = 'none' # normal, reverb, background, distort, or none
Cfg.p = 0.5 # for normal, background, and distort, probabilty of picking each augmentation