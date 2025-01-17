import random
from audiomentations import Compose, AddBackgroundNoise, AddGaussianNoise, AddGaussianSNR, ClippingDistortion
from audiomentations import FrequencyMask,PitchShift, PolarityInversion, Shift, TimeMask, TimeStretch, SpecFrequencyMask
from pysndfx import AudioEffectsChain
import numpy as np
import torch
import librosa

class Augmentor(object):
    def __init__(self, augment_type, p, cross_valid=False):
        self.cross_valid = cross_valid
        self.sample_rate = 8000
        self.type = augment_type
        self.p = p
        wham_path = '../../../librimix/data/wham_noise/cv' if self.cross_valid else '../../../librimix/data/wham_noise/tr'
        if self.type == 'wham_weak':
            self.augment = Compose([
                AddBackgroundNoise(sounds_path=wham_path, min_snr_in_db=5, max_snr_in_db=15, p=1)
            ])
        elif self.type == 'wham_strong':
            self.augment = Compose([
                AddBackgroundNoise(sounds_path=wham_path, min_snr_in_db=2, max_snr_in_db=7, p=1)
            ])
        elif self.type == 'reverb_weak':
            self.augment = AudioEffectsChain().reverb(reverberance=random.randrange(0, 50),
                                                      room_scale=random.randrange(0,50),
                                                      stereo_depth=random.randrange(0, 50),
                                                     )
        elif self.type == 'reverb_strong':
            self.augment = AudioEffectsChain().reverb(reverberance=random.randrange(50, 100),
                                                      room_scale=random.randrange(50, 100),
                                                      stereo_depth=random.randrange(50, 100),
                                                      )
        elif self.type == 'cascade':
            self.augment = Compose([
                AddBackgroundNoise(sounds_path=wham_path, min_snr_in_db=0, max_snr_in_db=5, p=self.p),
                AddGaussianSNR(min_SNR=0.001, max_SNR=0.25, p=self.p),
                ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=40, p=self.p),
                FrequencyMask(min_frequency_band=0.0, max_frequency_band=0.5, p=self.p),
                PolarityInversion(p=self.p),
                Shift(min_fraction=-0.5, max_fraction=0.5, rollover=True, p=self.p),
                TimeMask(min_band_part=0.0, max_band_part=0.2, fade=False, p=self.p)
            ])
        elif self.type == 'distort':
            self.augment = Compose([
                PitchShift(min_semitones=-4, max_semitones=4, p=self.p),
                TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=self.p)
            ])
        elif self.type == 'none':
            self.augment = None
        else:
            raise ValueError("Did not recognize augmentation type. Received %s, expected 'wham_weak', 'wham_strong', 'reverb_weak', 'reverb_strong', 'cascade', 'distort', or 'none'." % self.type)


    def augment_samples(self, sources):
        """ Transform the separate audio sources and return the augmented mixture and sources"""
        augment_sources = []
        if self.type == 'none':
            mix = sum(sources)
            return(mix, sources)
        elif 'reverb' in self.type:
            for i, s in enumerate(sources):
                transformed = torch.Tensor(self.augment(s.numpy()))
                augment_sources.append(transformed)
            self.reverb_randomize()
        elif 'wham' in self.type or 'cascade' in self.type or 'distort' in self.type: # weak or strong 'wham', 'cascade', 'distort'
            for i, s in enumerate(sources):
                transformed = torch.Tensor(self.augment(s.numpy(), sample_rate=self.sample_rate))
                augment_sources.append(transformed)
                if i == 0:
                    self.augment.freeze_parameters() # make sure all other sources gets the same augmentation
            self.augment.unfreeze_parameters() # lets the next augmentation be different
        else:
            raise ValueError("Did not recognize augmentation type. Received %s, expected 'wham_weak', 'wham_strong', 'reverb_weak', 'reverb_strong', 'cascade', 'distort', or 'none'." % self.type)
        augment_mix = sum(augment_sources)
        assert augment_mix.shape[0] == len(augment_sources[0]), "mixture length does not match length of sources, %i, %i" %  (augment_mix.shape[0], len(augment_sources[0]))
        return(augment_mix, augment_sources)

    def transform_speakers(self, speakers):
        """Used to transform the separate speakers, without the added noise"""
        speaker_augment = Compose([t for t in augment.transforms if 'noise' not in t.__str__().lower()])
        return(speaker_augment(speakers, sample_rate=self.sample_rate))

    def reverb_randomize(self):
        """Randomize parameters of reverberation augmentation"""
        self.augment = AudioEffectsChain().reverb(reverberance=random.randrange(50, 100),
                                                  room_scale=random.randrange(50, 100),
                                                  stereo_depth=random.randrange(50))
