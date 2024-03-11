import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import librosa
import librosa.display
import os
import glob
 
## 音声ファイルの読み込み
#sample_rate, samples = wavfile.read("example-data/01-01-08-1.wav")
# 
## FFTをかけて周波数スペクトルを取得
#frequencies, times, spec = spectrogram(samples, fs=sample_rate)
# 
## スペクトログラム画像の描画
#Z = 10. * np.log10(spec)
#plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
#plt.xlabel('Time (sec)')
#plt.ylabel('Frequency (Hz)')
#plt.colorbar()
#plt.show()

dir_path = "./test_data/"
file_list = glob.glob(os.path.join(dir_path, "*.wav"))
for wav_path in file_list:
	x, sr = librosa.load(wav_path, sr=44100)

	print(type(x), type(sr))
	print(x.shape, sr)

	X = librosa.stft(x)
	Xdb = librosa.amplitude_to_db(abs(X))
#plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#plt.colorbar()
#plt.show()

	plt.figure(figsize=(14, 5))
	librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
	plt.colorbar()
	#plt.show()
	plt.savefig(wav_path + ".png", format="png")
	#plt.savefig(wav_path + ".png", format="png", dpi=300)
