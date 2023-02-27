import numpy as np
from shakespeare_utils import vectorize
import h5py

data = open('data/shakespeare.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
chars = sorted(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

def build_data(data, T_x):
    X = []
    Y = []
    
    for i in range(T_x, len(data), 3):
        X.append(data[i - T_x:i])
        Y.append(data[i])
        
    return X, Y

T_x = 40
X, Y = build_data(data, T_x)

def vectorization(X, Y, char_to_ix, vocab_size, T_x):    
    x = np.zeros((len(X), T_x, vocab_size))
    y = np.zeros((len(Y), vocab_size))
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            character = X[i][j]
            ix_character = char_to_ix[character]
            x[i, j, ix_character] = 1
        character = Y[i]
        ix_character = char_to_ix[character]
        y[i, ix_character] = 1
    
    return x, y

X_train, y_train = vectorization(X, Y, char_to_ix, vocab_size, T_x)

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.models import load_model

classifier = load_model('model/model_shakespeare_kiank.h5')

classifier_2 = Sequential()
classifier_2.add(LSTM(units = 128, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
classifier_2.add(Dropout(0.5))
classifier_2.add(LSTM(units = 128, return_sequences = False))
classifier_2.add(Dropout(0.5))
classifier_2.add(Dense(units = X_train.shape[2], activation = 'softmax'))

classifier_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

classifier_2.fit(X_train, y_train, epochs = 100, batch_size = 32)

sentence = 'Forsooth this maketh no sense '.lower()

weights = classifier.get_weights()

def sample(preds, vocab_size, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(list(range(vocab_size)), p = probas.ravel())
    
    return out

def output(classifier, sentence, char_to_ix, vocab_size):
    x = vectorize(sentence, char_to_ix)
    x = np.transpose(x, axes = [1, 2, 0])
    indexes = []
    predictions = []
    for i in range(400):
        pred = classifier.predict(x, verbose = 0)[0]
        predictions.append(pred)
        next_idx = sample(pred, vocab_size)
        indexes.append(next_idx)
        x_pred = np.zeros((1, vocab_size))
        x_pred[0, next_idx] = 1
        x = np.squeeze(x)
        x = np.concatenate((x[1:], x_pred), axis = 0)
        x = np.expand_dims(x, axis = 0)
        
    return indexes, predictions

indexes, predictions = output(classifier_2, sentence, char_to_ix, vocab_size)

def print_prediction(sentence, indexes, ix_to_char):
    txt = ''.join(ix_to_char[idx] for idx in indexes)
    txt = sentence + txt
    print(txt, end = '')

print_prediction(sentence, indexes, ix_to_char)
