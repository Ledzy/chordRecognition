import numpy as np
import os
from utils import display
from reader import get_label, get_df, slice_sample, load_data
from model import LSTMModel
from keras.optimizers import Adam
from keras.utils import to_categorical


features, labels = load_data(root='./data/train')
valid_features, valid_labels = load_data(root='./data/test')
classes = list(set(np.hstack([labels.flatten(), valid_labels.flatten()])))
num_classes = len(classes)
for i, c in enumerate(classes):
    labels[labels==c] = i
    valid_labels[valid_labels==c] = i

labels = to_categorical(labels, num_classes=num_classes)
valid_labels = to_categorical(valid_labels, num_classes=num_classes)

index = np.random.permutation(features.shape[0])
features = features[index]
labels = labels[index]

model = LSTMModel(categories=num_classes)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=features, y=labels, validation_data=(valid_features, valid_labels), epochs=10, batch_size=3)