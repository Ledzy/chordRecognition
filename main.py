import librosa
import numpy as np
from utils import display
from model import LSTMModel

filename = '01_-_A_Hard_Day\'s_Night.wav'
y, sr = librosa.load(filename)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)