from easydict import EasyDict

Cfg = EasyDict()
Cfg.augment = 'normal' # normal, reverb or distort, TODO: implement spec_aug, background
Cfg.p = 0.5 # for normal and distort, probabilty of picking each augmentation