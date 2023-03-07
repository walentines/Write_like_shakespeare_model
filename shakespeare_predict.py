import keras
from keras.models import load_model
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

classifier_2 = load_model("model/model_shakespeare_kiank.h5")

sentence = input("Your Sentence: ").lower()

weights = classifier_2.get_weights()

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
