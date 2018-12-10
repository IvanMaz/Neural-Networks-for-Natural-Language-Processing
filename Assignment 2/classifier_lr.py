from math import log
import string
import re
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


def get_ngrams(tokens, ngram_range):
    return [" ".join(tokens[idx:idx+i])
            for i in range(ngram_range[0], ngram_range[1]+1)
            for idx in range(len(tokens)-i+1)]


def get_tokens(texts, ngram_range=(1, 1)):
    return [get_ngrams(tokenize(r), ngram_range) for r in texts]


def get_vocab(tokenized_train_texts):
    vocab = []

    for txt in tokenized_train_texts:
        vocab.extend([w for w in txt if w not in STOPWORDS])

    vocab = set(vocab)  # for faster searching
    print('размер словаря - %d' % len(vocab))
    return vocab


def get_matrix(tokenized_train_texts, vocab, shape, vocabulary={}):
    indptr = [0]
    indices = []
    data = []
    for d in tokenized_train_texts:
        for term in d:
            if term in vocab:
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
        indptr.append(len(indices))

    X = csr_matrix((data, indices, indptr), shape=shape, dtype=int)

    return X, vocabulary


class LogReg:
    def __init__(self, lr=1e-3, C=1e-3, epochs=1000, optim='adam'):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.optim = optim

        self.sparse = False
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        self.alpha = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

    def __sigmoid(self, z):
        return 1./(1+np.exp(-z))

    def __init_weights(self):
        self.w = np.zeros(self.n_features+1)
        self.m = np.zeros(self.n_features+1)
        self.v = np.zeros(self.n_features+1)

    def __h(self, x):
        return self.__sigmoid(x)

    def __loss(self, h, y):
        return -(y*np.log(h+1e-8)+(1-y)*np.log(1-h+1e-8)).mean() + (self.C*self.w**2).sum()

    def __grad(self, X, h, y):
        self.gradient = X.T.dot(h-y)/y.size + 2*self.C*self.w

    def __step(self):
        if self.optim == 'adam':
            self.t += 1

            self.m = self.beta1*self.m + (1-self.beta1)*self.gradient
            self.v = self.beta2*self.v + (1-self.beta2)*(self.gradient**2)

            self.m_cap = self.m/(1-(self.beta1**self.t))
            self.v_cap = self.v/(1-(self.beta2**self.t))

            self.w -= self.lr*self.m_cap/(np.sqrt(self.v_cap) + self.eps)
        elif self.optim == 'sgd':
            self.w -= self.lr*self.gradient

    def fit(self, X_train, y_train):
        self.n_samples, self.n_features = X_train.shape
        if issparse(X_train):
            self.sparse = True
        if self.sparse:
            X_train = hstack([np.ones((self.n_samples, 1)), X_train])
            #X = hstack((X,np.ones(self.n_samples)[:,None]))
        else:
            X_train = np.concatenate(
                (np.ones((self.n_samples, 1)), X_train), axis=1)

        self.__init_weights()
        for i in range(self.epochs):
            z = X_train.dot(self.w)
            h = self.__sigmoid(z)
            self.__grad(X_train, h, y_train)
            self.__step()

            if(i % 100 == 0):
                print(f'loss: {self.__loss(h, y_train)} \t')

    def predict_prob(self, X):
        self.n_samples, self.n_features = X.shape
        if issparse(X):
            self.sparse = True
        if self.sparse:
            X = hstack([np.ones((self.n_samples, 1)), X])

        return self.__sigmoid(X.dot(self.w))

    def predict(self, X):
        res = []
        for p in self.predict_prob(X).round():
            if p:
                res.append("pos")
            else:
                res.append("neg")
        return res

    def score(self, X, y):
        return (y == self.predict(X)).mean()


def train(train_texts, train_labels):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    ngram_range = (1, 1)
    tokenized_train_texts = get_tokens(train_texts, ngram_range)
    vocab = get_vocab(tokenized_train_texts)
    X_train, vocabulary = get_matrix(
        tokenized_train_texts, vocab, (len(tokenized_train_texts), len(vocab)))
    y_train = np.array([(lambda l: 1 if l == 'pos' else 0)(l)
                        for l in train_labels])

    model = LogReg(lr=1e-1, C=1e-3, epochs=10000, optim='sgd')
    model.fit(X_train, y_train)

    return model, vocab, vocabulary


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    model, vocab, vocabulary = params

    ngram_range = (1, 1)
    tokenized_texts = get_tokens(texts, ngram_range)

    X, _ = get_matrix(tokenized_texts, vocab,
                         (len(tokenized_texts), len(vocab)), vocabulary)
    preds = model.predict(X)

    return preds
