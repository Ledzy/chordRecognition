import keras
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed
from keras.losses import categorical_crossentropy

class LSTMModel(Model):
    """baseline LSTM model for classification
    params:
        categories: num of chord categories
        dropout: dropout ratio
    
    call:
        inputs: [batch, time_step, feature]    
     """

    def __init__(self, categories, dropout=0.):
        super(LSTMModel, self).__init__(name='lstm')
        self.lstm1 = LSTM(64, return_sequences=True)
        self.lstm2 = LSTM(32, return_sequences=True)
        self.dropout = Dropout(dropout)
        self.dense1 = Dense(categories, activation='softmax')
    
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout(x)
        x = self.lstm2(x)
        x = TimeDistributed(self.dense1)(x)
        return x