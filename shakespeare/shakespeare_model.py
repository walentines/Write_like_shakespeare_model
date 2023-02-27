import numpy as np
import shakespeare_utils as utils
from progress.bar import IncrementalBar

data = open('data/shakespeare.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
chars = sorted(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

sentences = ['Forsooth this maketh no sense ', 'm fairest creatures we desire increase,',  'A while ago I was home', 'Greetings exalted, Senator ', 'I am a lost soul ', 'This makes no sense ', 'Why am I doing this, ', 'Hello there are you here?', 'Help me, please ', 'I m driving insane', 'Greetings, I am a bold one right?', 'I am the chosen one ', 'I am rich what about you?', 'I m writing poems', 'This is the last one ']

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
    x = np.zeros((vocab_size, len(X), T_x))
    y = np.zeros((vocab_size, len(Y)))
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            character = X[i][j]
            ix_character = char_to_ix[character]
            x[ix_character, i, j] = 1
        character = Y[i]
        ix_character = char_to_ix[character]
        y[ix_character, i] = 1
    
    return x, y

X_train, y_train = vectorization(X, Y, char_to_ix, vocab_size, T_x)
 
def optimize(X_train, y_train, parameters, a_prev, a_prev1, learning_rate, v, s, t, keep_prob):
    a, y_pred, c, caches = utils.lstm_forward_2(X_train, a_prev, a_prev1, parameters, keep_prob = keep_prob)
    cost = utils.compute_cost(y_pred, y_train)
    da, dWy, dby = utils.backpropagation(y_pred, y_train, caches)
    gradients = utils.lstm_backward_2(da, dWy, dby, caches, keep_prob = keep_prob)
    parameters, v, s = utils.adam(parameters, gradients, learning_rate, v, s, t)
    
    return y_pred, cost, parameters, a[0][:, :, X_train.shape[2] - 1], a[1][:, :, X_train.shape[2] - 1], v, s

def model_2(X_train, y_train, ix_to_char, char_to_ix, learning_rate, num_epochs = 100, vocab_size = 38, n_a = 128):
    n_x, n_y = vocab_size, vocab_size
    m = X_train.shape[1]
    cnt = 0
    minibatch_size = 32
    
    parameters = utils.initialize_parameters(n_x, n_y, n_a)
    v, s = utils.initialize_adam(parameters)
    a_prev, a_prev1 = np.zeros((n_a, minibatch_size)), np.zeros((n_a, minibatch_size))

    for epoch in range(num_epochs):
        epoch_cost = 0.
        num_minibatches = m // minibatch_size
        minibatches = utils.random_mini_batches(X_train, y_train, mini_batch_size = minibatch_size)
        minibatches = minibatches[:-1]
        t = 0
        bar = IncrementalBar('Training', max = num_minibatches)
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            t += 1
            y_pred, cur_loss, parameters, a_prev, a_prev1, v, s = optimize(minibatch_X, minibatch_Y, parameters, a_prev, a_prev1, learning_rate, v, s, t, keep_prob = 0.5)
            epoch_cost += cur_loss
            bar.next()
        bar.finish()
        sentence = sentences[cnt].lower()
        x = utils.vectorize(sentence, char_to_ix)
        print()
        print('Iteration %d, Cost: %f' % (epoch, epoch_cost / num_minibatches) + '\n')
        print()
        sampled_indices = utils.sample(x, parameters, char_to_ix, temperature = 1.0)
        utils.print_sample(sentence, sampled_indices, ix_to_char)
        if cnt == 12:
            cnt = 0
        
    return parameters

parameters = model_2(X_train, y_train, ix_to_char, char_to_ix, learning_rate = 0.001)
