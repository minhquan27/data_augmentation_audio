import torch
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio, display


# Hàm đọc đầu vào là một audio đầu ra là audio_singal, sample_rate và time.
def load_audio(path: str):
    # input: audio path
    # output: audio_data: tensor, sample_rate: 48000, time_axis: time of audio
    audio_data, sample_rate = torchaudio.load(path)
    time_axis = audio_data.shape[1] / float(sample_rate)
    return audio_data, sample_rate, time_axis


# Hàm biểu diễn tín hiệu âm thanh trong miền thời gian
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


# Hàm hiển thị ảnh phổ của âm thanh (spectrogram)
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show()


# Hàm chuyển đổi tín hiệu âm thanh ra audio
def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


if __name__ == '__main__':
    audio = "/Users/nguyenquan/Desktop/vps/voice-gender-classifcation/data_test/03-01-01-01-01-01-23.wav"
    audio_data, sample_rate, time_axis = load_audio(audio)
    plot_waveform(audio_data, sample_rate)
    plot_specgram(audio_data, sample_rate)
