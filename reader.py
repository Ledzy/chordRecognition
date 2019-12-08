import pandas as pd
import numpy as np
import os
import pickle
import logging
import librosa
from io import StringIO

logging.basicConfig(level=logging.DEBUG)
with open('./label2idx.pkl', 'rb') as f:
    label2idx = pickle.load(f)

def get_df(path):
    origin_f = open(path, 'r')
    columns = 'start, end, label\n'
    content = columns + origin_f.read().replace(' ', ', ')
    df = pd.read_csv(StringIO(content), sep=', ')
    return df

def read_files(path):
    """create dict which helds chord information of files in dataframe format
    params:
        path: root path holding lab files
    return:
        df_dict: a dictionary mapping audio_name to DataFrame"""
    df_dict = {}
    for root, dirs, files in os.walk(path):
        for f in files:
            if './lab' in f:
                path = os.path.join(root, f)
                df_dict[f.strip('.lab')] = get_df(path)
    return df_dict

def get_label(df, chroma, label2idx, label_dim=407):
    """create one sample's label according to its df & chroma"""
    audio_len = chroma.shape[1] / 43.1 # the parameter is arbitrary, to be adjusted
    tag_len = df[-1:]['end'].values[0]
    zoom_factor = audio_len / tag_len
    df[['start', 'end']] *= zoom_factor

    label = np.zeros([chroma.shape[1], label_dim])
    for frame in df.itertuples():
        start = int(frame.start * zoom_factor * 43.1)
        end = int(frame.end * zoom_factor * 43.1)
        label_idx = label2idx[frame.label]
        label[start:end, label_idx] = 1
    return label

def slice_sample(feature, label, pieces=20, duration=30):
    """random cut samples into slices """
    feature = np.swapaxes(feature, 0, 1)
    assert feature.shape[1] == 12

    label = label.argmax(axis=1)
    break_points = np.where(label!=np.roll(label,1))[0]
    break_points = break_points[break_points < len(label)-np.ceil(duration * 43.1)]

    logging.debug(f'pieces:{pieces}, break_points:{len(break_points)}')
    start_idx = np.random.choice(break_points, min(pieces, len(break_points)), replace=False)
    end_idx = start_idx + int(duration*43.1)

    features = np.stack([feature[s_idx:e_idx] for s_idx, e_idx in zip(start_idx, end_idx)], 0)
    labels = np.stack([label[s_idx:e_idx] for s_idx, e_idx in zip(start_idx, end_idx)], 0)

    return features, labels

def get_audio2lab(root):
    #TODO
    audios = os.listdir(os.path.join(root, 'audio'))
    labs = os.listdir(os.path.join(root, 'lab'))
    audio2lab = {}
    for a, l in zip(audios, labs):
        audio2lab[a] = l
    return audio2lab

def load_data(root, pieces=20, duration=30):
    audio2lab = get_audio2lab(root)
    features, labels = [], []

    for a, l in audio2lab.items():
        y, sr = librosa.load(os.path.join(root, 'audio', a))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        df = get_df(os.path.join(root, 'lab', l))
        label = get_label(df, chroma, label2idx)
        pc_features, pc_labels = slice_sample(chroma, label, pieces=20)
        features.append(pc_features)
        labels.append(pc_labels)
    
    features = np.vstack(features)
    labels = np.vstack(labels)

    return features, labels


if __name__ == "__main__":
    source_dir = './The Beatles/chordlab/The Beatles'
    df_dict = {}
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if '.lab' in f:
                path = os.path.join(root, f)
                df_dict[f.strip('.lab')] = get_df(path)
                print(len(df_dict))

    combined_df = pd.concat(list(df_dict.values()))
    print(combined_df['label'].value_counts())