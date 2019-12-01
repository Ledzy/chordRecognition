import pickle
import librosa
import numpy as np
from utils import display
from reader import get_label, get_df, slice_sample
# from model import LSTMModel

filename = '07 - Please Please Me.flac' #2:07 127
# filename = '03 - Anna.flac' #2:59 179

y, sr = librosa.load(filename)
chroma = librosa.feature.chroma_stft(y=y, sr=sr) 

with open('./label2idx.pkl', 'rb') as f:
    label2idx = pickle.load(f)
df = get_df('07_-_Please_Please_Me.lab')
label = get_label(df, chroma, label2idx)

features, labels = slice_sample(chroma, label)