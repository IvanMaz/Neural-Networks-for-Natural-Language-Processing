from math import log
import string
import re
import os
import sys
import random
import numpy as np
from statistics import mean, median
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, hstack, issparse, coo_matrix


punct = string.punctuation + '…–—‘“‚„«»'
STOPWORDS = {'now', 'up', 'against', 'again', 'from', 'or', 'did', 'here', 'his', 'those', 'had',
             'few', 'own', 'ma', 'all', 'where', 's', 'are', 'being', 'herself', 'ourselves',
             'themselves', 'and', 'at', 'can', 'i', 'I', 'why', 'ours', 'they', 'wasn', 'be', 'hasn',
             'doing', 'am', 'when', 'didn', 'them', 'a', 'o', 'same', 'nor', 'won', 'will', 'while', 'm',
             'isn', 'shouldn', 'which', 'has', 'hadn', 'than', 'mustn', 'off', 'each', 'is', 'y',
             'himself', 'should', 'hers', 'more', 'some', 'until', 'other', 'most', 'of', 'by', 'whom',
             'these', 'after', 'aren', 'their', 'too', 'couldn', 'the', 'me', 'yourself', 'do', 'shan',
             'weren', 'your', 'if', 'he', 'in', 'it', 'our', 'who', 'd', 'as', 'any', 'yourselves',
             'having', 'an', 'she', 'my', 'haven', 'been', 'there', 'such', 'her', 'because', 'how',
             'myself', 'theirs', 'only', 'with', 'itself', 'before', 'its', 'once', 'that', 'about',
             'out', 'so', 'have', 'but', 'on', 'we', 're', 'further', 'during', 'for', 've', 'very',
             'you', 'then', 'this', 'over', 'ain', 'between', 'under', 'not', 'yours', 'does', 'both',
             'him', 'what', 'above', 'were', 'below', 'just', 'into', 'no', 'wouldn', 'needn', 'doesn',
             'to', 'down', 'mightn', 't', 'through', 'was', 'll'}


def tokenize(text):
    """
    Preprocesses text and split it into the list of words
    :param: text(str): movie review
    """
    return re.findall(r'\w+', text.lower())


def get_tokens(texts):
    return [tokenize(r) for r in texts]

def clean_and_vectorize(text, word2idx, vocab, glove_vec):
    text = [clean(r, word2idx, vocab) for r in text]
    return np.array([vectorize(r, glove_vec, word2idx) for r in text])

def get_vocab(tokenized_train_texts):
    vocab = []

    for txt in tokenized_train_texts:
        vocab.extend([w for w in txt if w not in STOPWORDS])

    vocab = set(vocab)  # for faster searching
    print('размер словаря - %d' % len(vocab))
    return vocab

def load_glove(dim=50):
    word2idx = {}
    idx2word = {}
    vecs = []
    file_name = 'glove.6B.{}d.txt'.format(dim)
    glove_path = os.path.join('../GLOVE', file_name)
    for idx, line in enumerate(open(glove_path, 'r', encoding='utf8')):
        word, vec = line.split(' ', 1)
        word2idx[word] = idx
        idx2word[idx] = word
        vecs.append(np.array(list(map(float, vec.split()))))
    return word2idx, np.array(vecs)

def clean(txt, word2idx, vocab):
    return [word for word in txt if word in word2idx and word in vocab]

def vectorize(txt, glove_vec, word2idx):
    if len(txt):
        return sum([glove_vec[word2idx[word]] for word in txt]) / len(txt)
    else:
        return np.zeros(glove_vec.shape[1])


class SGD:
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self, grads, learning_rate):
        for i in range(len(grads)):
            self.parameters[i] -= learning_rate * grads[i]

class Linear:
    def __init__(self, input_size, output_size, alpha):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.b = np.zeros(output_size)
        self.alpha = alpha

        self.optimizer = SGD([self.W, self.b])

    def forward(self, X):
        self.X = X
        return X.dot(self.W)+self.b

    def backward(self, dLdy):
        self.dLdW = self.X.T.dot(dLdy) + self.alpha*self.W
        self.dLdb = dLdy.sum(0)
        self.dLdx = dLdy.dot(self.W.T)
        return self.dLdx

    def step(self, learning_rate):
        self.optimizer.step([self.dLdW, self.dLdb], learning_rate)

class ReLU:
    def __init__(self):
        pass
    
    def forward(self, X):
        self.X = X
        return X.clip(min=0)
    
    def backward(self, dLdy):
        return dLdy * (np.heaviside(self.X, 0))
    
    def step(self, learning_rate):
        pass


class NLLLoss:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def forward(self, X, y, modules):
        self.p = np.exp(X)
        self.p /= self.p.sum(1, keepdims=True)
        self.y = np.zeros((X.shape[0], X.shape[1]))
        self.y[np.arange(X.shape[0]), y] = 1
        self.X = X
        
        L2 = 0
        for m in modules:
            if m.__class__.__name__ == "Linear":
                L2 += self.alpha*(np.sum(np.square(m.W)))
        
        return -(np.log(self.p)*self.y).sum(1).mean(0) + L2
    
    def backward(self):        
        return (self.p - self.y) / self.X.shape[0]

class NeuralNetwork:
    def __init__(self, modules):
        self.modules = modules
    
    def forward(self, X):
        for m in self.modules:
            X = m.forward(X)
        
        return X, self.modules
    
    def backward(self, dLdy):
        for m in self.modules[::-1]:
            dLdy = m.backward(dLdy)
        
    
    def step(self, learning_rate):
        for m in self.modules:
            m.step(learning_rate)

def train_(nn, Xtrain, Ytrain, learning_rate, num_epochs, batch_size, alpha):
    
    train_loss = []
    train_acc = []
    
    for epoch in range(num_epochs):
        nll = NLLLoss(alpha)
        combined = list(zip(Xtrain, Ytrain))
        random.shuffle(combined)

        Xtrain_shuf, Ytrain_shuf = zip(*combined)
        Xtrain_shuf = np.array(Xtrain_shuf)
        Ytrain_shuf = np.array(Ytrain_shuf)
        
        losses = []
        accs = []
        for idx in range(0, len(Xtrain_shuf), batch_size):
            y_, modules = nn.forward(Xtrain_shuf[idx:idx+batch_size])
            
            loss = nll.forward(y_, Ytrain_shuf[idx:idx+batch_size], modules)
            acc = np.mean(np.argmax(y_, axis=1) == Ytrain_shuf[idx:idx+batch_size])

            accs.append(acc)
            losses.append(loss)

            dLdy = nll.backward()
            
            nn.backward(dLdy)
            nn.step(learning_rate)
        
        train_loss.append(np.mean(losses))
        train_acc.append(np.mean(accs))
        print(epoch, train_loss[-1], train_acc[-1])
    return

def train(train_texts, train_labels):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    word2idx, glove_vec = load_glove(300)

    tokenized_train_texts = get_tokens(train_texts)
    vocab = get_vocab(tokenized_train_texts)

    X_train = clean_and_vectorize(tokenized_train_texts, word2idx, vocab, glove_vec)
    y_train = np.array([(lambda l: 1 if l == 'pos' else 0)(l)
                        for l in train_labels])

    alpha = 1e-4
    lr = 0.01
    epochs = 150
    batch_size = 128
    model = NeuralNetwork([
                    Linear(300, 100, alpha),
                    ReLU(),
                    Linear(100, 2, alpha),
                ])
    train_(model, X_train, y_train, 
            lr, epochs, batch_size, alpha)

    return model, vocab, word2idx, glove_vec


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    model, vocab, word2idx, glove_vec = params

    tokenized_texts = get_tokens(texts)

    X_train = clean_and_vectorize(tokenized_texts, word2idx, vocab, glove_vec)

    preds = np.argmax(model.forward(X_train)[0], 1)
    preds = np.array([(lambda l: 'pos' if l == 1 else 'neg')(l)
                        for l in preds])

    return preds
