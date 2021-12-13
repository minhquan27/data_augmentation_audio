import random
import torch
import torchaudio
import os
import pathlib
import math


class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length
        self.vad = torchaudio.transforms.Vad(sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_data = self.vad(audio_data)
        audio_length = audio_data.shape[1]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio_data = audio_data[:, offset: (offset + self.clip_length)]
        return audio_data


class RandomSpeedChange:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data, speed_factor=1.1):
        # speed_factor = [0.9, 1.0, 1.1]
        if speed_factor == 1.0:  # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio


# noise dir: http://www.openslr.org/17/
class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.noise_dir = noise_dir
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(self.noise_dir):
            raise IOError(f'Noise directory {self.noise_dir} does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(self.noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory {self.noise_dir}')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'],  # convert to mono
            ['rate', str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset:offset + audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise) / 2


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data


def feature_extractions(audio):
    audio_signal, sample_rate = torchaudio.load(audio)
    transform = torchaudio.transforms.Spectrogram()
    spectrogram = transform(audio_signal)
    transforms = torchaudio.transforms.MFCC()
    mfcc = transforms(audio_signal)
    return spectrogram, mfcc


if __name__ == '__main__':
    audio = "/Users/nguyenquan/Desktop/vps/voice-gender-classifcation/data_test/03-01-01-01-01-01-23.wav"
    spectrogram, mfcc = feature_extractions(audio)