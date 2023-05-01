import glob
import librosa
import numpy as np
import soundfile as sf
from os.path import exists
from multiprocessing import Pool
from sklearn.preprocessing import MaxAbsScaler

frame_length=0.05
frame_shift=0.0125
n_fft=2048
target_sample_rate=24000
STFT_len = 16000
sample_len = 8000

def quantize(S):
    scaler = MaxAbsScaler()
    S_scaled = scaler.fit_transform(S)
    S_mu = librosa.mu_compress(S_scaled, mu=15, quantize=True)
    S_quant = librosa.mu_expand(S_mu, mu=15, quantize=True)
    return scaler.inverse_transform(S_quant)

def slice(y, n):
    if len(y) > n:
        start_index = (len(y) - n) // 2
        y_hat = y[start_index : start_index + n]
        return y_hat
    else:
        return y

def convert_n_export(filepath, folder='Emo-Soundscapes'):
    file_name = filepath.split('/')[-1]
    new_path = f"./data/reconstructed/{folder}/GL_" + file_name
    if (exists(new_path.replace('.mp3', '.wav'))):
        return
    
    y, sr = librosa.load(filepath, sr=target_sample_rate, mono=True)
    y = slice(y, STFT_len)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=int(frame_shift * sr), win_length=int(frame_length * sr), window='hann'))
    S = quantize(S)

    y_inv = librosa.griffinlim(S, n_iter=100, n_fft=n_fft, hop_length=int(frame_shift * sr), win_length=int(frame_length * sr), window='hann')
    y_inv = slice(y_inv, sample_len)
    print(f"Writing out to {new_path}......")
    sf.write(new_path.replace('.mp3', '.wav'), y_inv, sr, subtype='PCM_16')

if __name__ == '__main__':
    soundscapes_path = './data/raw/Emo-Soundscapes/Emo-Soundscapes-Audio/600_Sounds'
    soundscapes = list(glob.glob(f"{soundscapes_path}/*/**.wav"))

    envs_path = './data/raw/environmental-sound'
    envs = list(glob.glob(f"{envs_path}/*/**.wav"))

    loops_path = './data/raw/FSL10K/audio'
    loops = list(glob.glob(f"{loops_path}/*/**.wav"))

    voices_path = './data/raw/CommonVoice'
    voices = sorted(list(glob.glob(f"{voices_path}/cv-valid-train/*/**.mp3")))

    pool = Pool(processes=10)
    pool.map(convert_n_export, soundscapes)