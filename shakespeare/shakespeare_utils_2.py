import numpy as np
import math

def sigmoid(z):
    A = 1 / (1 + np.exp(-z))
    
    return A

def softmax(x):
    e_x = np.exp(x - np.max(x))
    
    return e_x / np.sum(e_x, axis = 0)

def dropout(A, keep_prob = 0.5):
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(int)
    A = A * D
    A = A / keep_prob
    dropout_cache = D
    
    return A, dropout_cache

def dropout_backward(dA, dropout_cache, keep_prob = 0.5):
    D = dropout_cache
    dA = dA * D
    dA = dA / keep_prob
    
    return dA

def clip(gradients, maxValue):
    (gradients, gradients1) = gradients
    
    dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby = gradients['dWf'], gradients['dbf'], gradients['dWi'], gradients['dbi'], gradients['dWc'], gradients['dbc'], gradients['dWo'], gradients['dbo'], gradients['dWy'], gradients['dby']
    
    for gradient in [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo, dWy, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient)
        
    gradients = {'dWf': dWf, 'dbf': dbf, 'dWi': dWi, 'dbi': dbi, 'dWc': dWc, 'dbc': dbc, 'dWo': dWo, 'dbo': dbo, 'dWy': dWy, 'dby': dby}
    
    return gradients

def get_initial_loss(vocab_size, seq_length):
    return - np.log(1.0 / vocab_size) * seq_length

def smooth(loss, cur_loss):
    return 0.999 * loss + 0.001 * cur_loss

def vectorize(string, char_to_ix):
    vocab_size = len(char_to_ix)
    T_x = len(string)
    x = np.zeros((vocab_size, 1, T_x))
    for i in range(T_x):
        idx = char_to_ix[string[i]]
        x[idx, :, i] = 1
    
    return x

def sample(x, parameters, char_to_ix, temperature):
    Wf = parameters['Wf']
    Wy = parameters['Wy']
    by = parameters['by']
    
    vocab_size = x.shape[0]
    n_a = Wf.shape[0]
    
    a_next = np.zeros((n_a, 1))
    
    indices = []
    idx = -1
    
    for i in range(400):
        a, y, _, _ = lstm_forward(x, a_next, parameters, keep_prob = 1.0)
        a_next = a[:, :, -1]
        y = softmax(np.dot(Wy, a_next) + by)
        y = np.squeeze(y)
        y = np.log(y) / temperature
        exp_y = np.exp(y)
        y = exp_y / np.sum(exp_y)
        probas = np.random.multinomial(1, y, 1)
        idx = np.random.choice(list(range(vocab_size)), p = probas.ravel())
        indices.append(idx)
        
        x_pred = np.zeros((vocab_size, 1, 1))
        x_pred[idx, :, 0] = 1
        x = np.concatenate((x[:, :, 1:], x_pred), axis = 2)
        
    return indices

def print_sample(sentence, sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = sentence[0].upper() + sentence[1:] + txt
    print('%s' % (txt, ), end = '')
    
def initialize_parameters(n_x, n_y, n_a):
    Wf = np.random.randn(n_a, n_a + n_x) * 0.01
    bf = np.zeros((n_a, 1))
    Wi = np.random.randn(n_a, n_a + n_x) * 0.01
    bi = np.zeros((n_a, 1))
    Wc = np.random.randn(n_a, n_a + n_x) * 0.01
    bc = np.zeros((n_a, 1))
    Wo = np.random.randn(n_a, n_a + n_x) * 0.01
    bo = np.zeros((n_a, 1))
    Wy = np.random.randn(n_y, n_a) * np.sqrt(2. / n_a)
    by = np.zeros((n_y, 1))
    
    parameters = {'Wf': Wf,
                   'bf': bf,
                   'Wi': Wi,
                   'bi': bi,
                   'Wc': Wc,
                   'bc': bc,
                   'Wo': Wo,
                   'bo': bo,
                   'Wy': Wy,
                   'by': by}
        
    return parameters

def initialize_adam(parameters):    
    v = {}
    s = {}
    
    v['dWf'] = np.zeros((parameters['Wf'].shape[0], parameters['Wf'].shape[1]))
    v['dbf'] = np.zeros((parameters['bf'].shape[0], parameters['bf'].shape[1]))
    v['dWi'] = np.zeros((parameters['Wi'].shape[0], parameters['Wi'].shape[1]))
    v['dbi'] = np.zeros((parameters['bi'].shape[0], parameters['bi'].shape[1]))
    v['dWc'] = np.zeros((parameters['Wc'].shape[0], parameters['Wc'].shape[1]))
    v['dbc'] = np.zeros((parameters['bc'].shape[0], parameters['bc'].shape[1]))
    v['dWo'] = np.zeros((parameters['Wo'].shape[0], parameters['Wo'].shape[1]))
    v['dbo'] = np.zeros((parameters['bo'].shape[0], parameters['bo'].shape[1]))
    
    s['dWf'] = np.zeros((parameters['Wf'].shape[0], parameters['Wf'].shape[1]))
    s['dbf'] = np.zeros((parameters['bf'].shape[0], parameters['bf'].shape[1]))
    s['dWi'] = np.zeros((parameters['Wi'].shape[0], parameters['Wi'].shape[1]))
    s['dbi'] = np.zeros((parameters['bi'].shape[0], parameters['bi'].shape[1]))
    s['dWc'] = np.zeros((parameters['Wc'].shape[0], parameters['Wc'].shape[1]))
    s['dbc'] = np.zeros((parameters['bc'].shape[0], parameters['bc'].shape[1]))
    s['dWo'] = np.zeros((parameters['Wo'].shape[0], parameters['Wo'].shape[1]))
    s['dbo'] = np.zeros((parameters['bo'].shape[0], parameters['bo'].shape[1]))
    
    return v, s 

def random_mini_batches(X, Y, mini_batch_size = 256):
    m = X.shape[1]
    mini_batches = []
    
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[:, k * mini_batch_size : (1 + k) * mini_batch_size, :]
        mini_batch_Y = Y[:, k * mini_batch_size : (1 + k) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Last case
    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches 

def lstm_cell_forward(xt, a_prev, c_prev, parameters, keep_prob):
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    
    n_x, m = xt.shape

    concat = np.concatenate((a_prev, xt), axis = 0)
    
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = sigmoid(np.dot(Wc, concat) + bc)
    c_next = ft * c_prev + it * cct
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)
    a_next, dropout_cache = dropout(a_next, keep_prob = keep_prob)
        
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters, dropout_cache)
    
    return a_next, c_next, cache

def lstm_forward(x, a0, parameters, keep_prob):
    caches = []
    
    Wy = parameters['Wy']
    by = parameters['by']
    
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    a_next = a0
    c_next = np.zeros((n_a, m))
    for t in range(T_x):
        xt = x[:, :, t]
        a_next, c_next, cache = lstm_cell_forward(xt, a_next, c_next, parameters, keep_prob)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        caches.append(cache)
    z = np.dot(Wy, a_next) + by
    y = softmax(z)
    caches = (caches, x, z)
    
    return a, y, c, caches

def compute_cost(y_pred, y_train):
    m = y_train.shape[1]
    cost = - 1 / m * np.sum(y_train * np.log(y_pred))
    cost = float(cost)
    
    return cost

def backpropagation(y_pred, y_train, caches):
    (caches, x, z) = caches
    m = y_train.shape[1]
    T_x = x.shape[2]
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters, dropout_cache) = caches[T_x - 1]
    
    dz = y_pred - y_train
    
    dWy = 1 / m * np.dot(dz, a_next.T)
    dby = 1 / m * np.sum(dz, axis = 1, keepdims = True)
    
    da = np.dot(parameters['Wy'].T, dz)
    
    return da, dWy, dby

def lstm_cell_backward(da_next, dc_next, cache, keep_prob):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters, dropout_cache) = cache
    
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next) * (1 - cct ** 2)
    dit = (dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next) * ft * (1 - ft)
    
    dWf = np.dot(dft, np.concatenate((a_prev, xt), axis = 0).T)
    dWi = np.dot(dit, np.concatenate((a_prev, xt), axis = 0).T)
    dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis = 0).T)
    dWo = np.dot(dot, np.concatenate((a_prev, xt), axis = 0).T)
    dbf = np.sum(dft, axis = 1, keepdims = True)
    dbi = np.sum(dit, axis = 1, keepdims = True)
    dbc = np.sum(dcct, axis = 1, keepdims = True)
    dbo = np.sum(dot, axis = 1, keepdims = True)
    
    da_prev = np.dot(parameters['Wf'][:, :n_a].T, dft) + np.dot(parameters['Wi'][:, :n_a].T, dit) + np.dot(parameters['Wc'][:, :n_a].T, dcct) + np.dot(parameters['Wo'][:, :n_a].T, dot)
    da_prev = dropout_backward(da_prev, dropout_cache, keep_prob = keep_prob)
    dc_prev = dc_next * ft + ot * (1 - np.tanh(c_next) ** 2) * ft * da_next
    dxt = np.dot(parameters['Wf'][:, n_a:].T, dft) + np.dot(parameters['Wi'][:, n_a:].T, dit) + np.dot(parameters['Wc'][:, n_a:].T, dcct) + np.dot(parameters['Wo'][:, n_a:].T, dot)
    
    gradients = {'dxt': dxt,
                 'da_prevt': da_prev,
                 'dc_prevt': dc_prev,
                 'dWf': dWf,
                 'dbf': dbf,
                 'dWi': dWi,
                 'dbi': dbi,
                 'dWc': dWc,
                 'dbc': dbc,
                 'dWo': dWo,
                 'dbo': dbo}
    
    return gradients

def lstm_backward(da, dWy, dby, caches, keep_prob):    
    (caches, x, z) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters, dropout_cache) = caches[0]
    
    n_a, m = da.shape
    n_x, m = x1.shape
    T_x = x.shape[2]
    
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))
    
    gradients = lstm_cell_backward(da + da_prevt, dc_prevt, caches[T_x - 1], keep_prob)
    for t in reversed(range(T_x - 1)):
        da_prevt = gradients['da_prevt']
        dc_prevt = gradients['dc_prevt']
        dx[:, :, t] = gradients['dxt']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']
        gradients = lstm_cell_backward(da_prevt, dc_prevt, caches[t], keep_prob)
    
    dWf /= m
    dWi /= m
    dWc /= m
    dWo /= m
    dbf /= m
    dbi /= m
    dbc /= m
    dbo /= m
    
    da0 = da_prevt
    
    gradients = {'dx': dx,
                 'da0': da0,
                 'dWf': dWf,
                 'dbf': dbf,
                 'dWi': dWi,
                 'dbi': dbi,
                 'dWc': dWc,
                 'dbc': dbc,
                 'dWo': dWo,
                 'dbo': dbo,
                 'dWy': dWy,
                 'dby': dby}
    
    return gradients

def adam(parameters, gradients, learning_rate, v, s, t, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7):
    v_corrected = {}
    s_corrected = {}
    
    v['dWf'] = beta1 * v['dWf'] + (1 - beta1) * gradients['dWf']
    v['dbf'] = beta1 * v['dbf'] + (1 - beta1) * gradients['dbf']
    v['dWi'] = beta1 * v['dWi'] + (1 - beta1) * gradients['dWi']
    v['dbi'] = beta1 * v['dbi'] + (1 - beta1) * gradients['dbi']
    v['dWc'] = beta1 * v['dWc'] + (1 - beta1) * gradients['dWc']
    v['dbc'] = beta1 * v['dbc'] + (1 - beta1) * gradients['dbc']
    v['dWo'] = beta1 * v['dWo'] + (1 - beta1) * gradients['dWo']
    v['dbo'] = beta1 * v['dbo'] + (1 - beta1) * gradients['dbo']
    
    v_corrected['dWf'] = v['dWf'] / (1 - beta1 ** t)
    v_corrected['dbf'] = v['dbf'] / (1 - beta1 ** t)
    v_corrected['dWi'] = v['dWi'] / (1 - beta1 ** t)
    v_corrected['dbi'] = v['dbi'] / (1 - beta1 ** t)
    v_corrected['dWc'] = v['dWc'] / (1 - beta1 ** t)
    v_corrected['dbc'] = v['dbc'] / (1 - beta1 ** t)
    v_corrected['dWo'] = v['dWo'] / (1 - beta1 ** t)
    v_corrected['dbo'] = v['dbo'] / (1 - beta1 ** t)
    
    s['dWf'] = beta2 * s['dWf'] + (1 - beta2) * (gradients['dWf'] ** 2)
    s['dbf'] = beta2 * s['dbf'] + (1 - beta2) * (gradients['dbf'] ** 2)
    s['dWi'] = beta2 * s['dWi'] + (1 - beta2) * (gradients['dWi'] ** 2)
    s['dbi'] = beta2 * s['dbi'] + (1 - beta2) * (gradients['dbi'] ** 2)
    s['dWc'] = beta2 * s['dWc'] + (1 - beta2) * (gradients['dWc'] ** 2)
    s['dbc'] = beta2 * s['dbc'] + (1 - beta2) * (gradients['dbc'] ** 2)
    s['dWo'] = beta2 * s['dWo'] + (1 - beta2) * (gradients['dWo'] ** 2)
    s['dbo'] = beta2 * s['dbo'] + (1 - beta2) * (gradients['dbo'] ** 2)
    
    s_corrected['dWf'] = s['dWf'] / (1 - beta2 ** t)
    s_corrected['dbf'] = s['dbf'] / (1 - beta2 ** t)
    s_corrected['dWi'] = s['dWi'] / (1 - beta2 ** t)
    s_corrected['dbi'] = s['dbi'] / (1 - beta2 ** t)
    s_corrected['dWc'] = s['dWc'] / (1 - beta2 ** t)
    s_corrected['dbc'] = s['dbc'] / (1 - beta2 ** t)
    s_corrected['dWo'] = s['dWo'] / (1 - beta2 ** t)
    s_corrected['dbo'] = s['dbo'] / (1 - beta2 ** t)
    
    parameters['Wf'] = parameters['Wf'] - np.dot(learning_rate, v_corrected['dWf'] / (np.sqrt(s_corrected['dWf']) + epsilon))
    parameters['bf'] = parameters['bf'] - np.dot(learning_rate, v_corrected['dbf'] / (np.sqrt(s_corrected['dbf']) + epsilon))
    parameters['Wi'] = parameters['Wi'] - np.dot(learning_rate, v_corrected['dWi'] / (np.sqrt(s_corrected['dWi']) + epsilon))
    parameters['bi'] = parameters['bi'] - np.dot(learning_rate, v_corrected['dbi'] / (np.sqrt(s_corrected['dbi']) + epsilon))
    parameters['Wc'] = parameters['Wc'] - np.dot(learning_rate, v_corrected['dWc'] / (np.sqrt(s_corrected['dWc']) + epsilon))
    parameters['bc'] = parameters['bc'] - np.dot(learning_rate, v_corrected['dbc'] / (np.sqrt(s_corrected['dbc']) + epsilon))
    parameters['Wo'] = parameters['Wo'] - np.dot(learning_rate, v_corrected['dWo'] / (np.sqrt(s_corrected['dWo']) + epsilon))
    parameters['bo'] = parameters['bo'] - np.dot(learning_rate, v_corrected['dbo'] / (np.sqrt(s_corrected['dbo']) + epsilon))
    
    return parameters, v, s