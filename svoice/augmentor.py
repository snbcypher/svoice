from audiomentations import Compose, AddBackgroundNoise, AddGaussianNoise, AddGaussianSNR, ClippingDistortion
from audiomentations import FrequencyMask,PitchShift, PolarityInversion, Shift, TimeMask, TimeStretch, SpecFrequencyMask
import librosa

class Augmentor(object):
    def __init__(self, cross_valid=False):
        self.cross_valid = cross_valid

        self.SAMPLE_RATE = 8000

        wham_path = '../../librimix/data/wham_noise/cv' if self.cross_valid else '../../librimix/data/wham_noise/tr'
        self.augment = Compose([
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            AddBackgroundNoise(sounds_path=wham_path, p=0.5),
            ClippingDistortion(),
            FrequencyMask(),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            PolarityInversion(),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            TimeMask(),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
        ])

    def augment_samples(self, samples):
        """ Transform the mixed audio to augment data"""
        return(self.augment(samples, sample_rate=self.SAMPLE_RATE))

    def transform_speakers(self, speakers):
        """Used to transform the separate speakers, make sure not to add noise"""
        speaker_augment = Compose([t for t in augment.transforms if 'noise' not in t.__str__().lower()])

        return(speaker_augment(speakers, sample_rate=self.SAMPLE_RATE))
