import numpy as np
from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, LSTM, GRU
from keras.layers import Embedding
from keras.layers import Dense, Dropout, Input, BatchNormalization, Concatenate


def load_data(max_sequence_length, shuffle=False):
    (X_train, y_train), (X_test, y_test) = reuters.load_data()

    X_train = sequence.pad_sequences(X_train, maxlen=max_sequence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sequence_length)

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    if shuffle:
        shuffle_idx = np.random.permutation(range(X.shape[0]))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        X_train = X[:X_train.shape[0]]
        y_train = y[:X_train.shape[0]]
        X_test = X[X_train.shape[0]:]
        y_test = y[X_train.shape[0]:]

    y_train = to_categorical(y_train, 46)
    y_test = to_categorical(y_test, 46)

    class Data:
        def __init__(self):
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None
            self.max_features = None

    data = Data()
    data.X_train = X_train
    data.y_train = y_train
    data.X_test = X_test
    data.y_test = y_test
    data.max_features = np.max(X)
    return data


def build_lstm_model(max_features, max_len):
    model = Sequential()
    model.add(Embedding(max_features+1, output_dim=128, input_length=max_len))
    model.add(BatchNormalization())
    model.add(LSTM(128, recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(46, activation='sigmoid'))
    return model


def build_bidirectional_lstm_model(max_features, max_len):
    sequence = Input(shape=(max_len,), dtype='int32')
    embedded = Embedding(max_features+1, 128, input_length=max_len)(sequence)
    bnorm = BatchNormalization()(embedded)
    forwards = LSTM(128, recurrent_dropout=0.4)(bnorm)
    backwards = LSTM(128, recurrent_dropout=0.4, go_backwards=True)(bnorm)
    merged = Concatenate()([forwards, backwards])
    dropout = Dropout(0.5)(merged)
    output = Dense(46, activation='sigmoid')(dropout)
    model = Model(inputs=sequence, outputs=output)
    return model


def build_gru_model(max_features, max_len):
    model = Sequential()
    model.add(Embedding(max_features+1, output_dim=128, input_length=max_len))
    model.add(GRU(128, recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(46, activation='sigmoid'))
    return model


if __name__ == '__main__':
    load_data(200)
