from os import listdir
from math import log
import string
import re
from statistics import mean, median
from collections import Counter, defaultdict


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


def count(pos_tok, neg_tok, nb='ber', ngram_range=(1, 1)):

    # returns dicts pos_idx and neg_idx that match token and number of occurrences
    # dicts are already smoothed
    pos_idx = defaultdict(lambda: 1)
    neg_idx = defaultdict(lambda: 1)

    if nb == 'ber':
        # сount only one occurrence of token in review for Bernoulli Naive Bayes
        for rev in pos_tok:
            for token in set(get_ngrams(rev, ngram_range)):
                if token not in STOPWORDS:
                    pos_idx[token] += 1

        for rev in neg_tok:
            for token in set(get_ngrams(rev, ngram_range)):
                if token not in STOPWORDS:
                    neg_idx[token] += 1

    elif nb == 'mul':
        # count every occurance of token for Multinomial Naive Bayes
        for rev in pos_tok:
            for token in get_ngrams(rev, ngram_range):
                if token not in STOPWORDS:
                    pos_idx[token] += 1

        for rev in neg_tok:
            for token in get_ngrams(rev, ngram_range):
                if token not in STOPWORDS:
                    neg_idx[token] += 1
    else:
        print("Error")

    return pos_idx, neg_idx


def get_ngrams(tokens, ngram_range):
    return [" ".join(tokens[idx:idx+i])
            for i in range(ngram_range[0], ngram_range[1]+1)
            for idx in range(len(tokens)-i+1)]


def train(train_texts, train_labels):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    train_pos = []
    train_neg = []

    for s, l in zip(train_texts, train_labels):
        if l == 'neg':
            train_pos.append(s)
        elif l == 'pos':
            train_neg.append(s)
        else:
            return None

    train_pos_tok = [tokenize(r) for r in train_pos]
    train_neg_tok = [tokenize(r) for r in train_neg]

    pos_idx, neg_idx = map(Counter, count(train_pos_tok, train_neg_tok, 'ber', (2, 2)))

    train_pos_total, train_neg_total = len(train_pos_tok), len(train_neg_tok)
    dict_total = train_pos_total + train_neg_total
    
    pos_probs = defaultdict(lambda: 1/(train_pos_total+2))
    neg_probs = defaultdict(lambda: 1/(train_neg_total+2))

    for k, v in pos_idx.items():
        pos_probs[k] = v/(train_pos_total+2)

    for k, v in neg_idx.items():
        neg_probs[k] = v/(train_neg_total+2)

    unique_words = sorted(list(set(pos_probs.keys()) | set(neg_probs.keys())))

    trained_data = {k: [log(pos_probs[k]),
                        log(1-pos_probs[k]),
                        log(neg_probs[k]),
                        log(1-neg_probs[k])] for k in unique_words}

    pos_log_prior = log(train_pos_total) - log(dict_total)
    neg_log_prior = log(train_neg_total) - log(dict_total)

    # precoumputing negative probabilities for faster computation of Bernoulli Naive Bayes
    prior_pos_proba = sum([trained_data[k][1] for k in trained_data])
    prior_neg_proba = sum([trained_data[k][3] for k in trained_data])

    return pos_log_prior, neg_log_prior, prior_pos_proba, prior_neg_proba, trained_data


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    pos_log_prior, neg_log_prior, prior_pos_proba, prior_neg_proba, trained_data = params

    texts_tok = [tokenize(r) for r in texts]

    preds = []
    for text in texts_tok:
        pred_pos = prior_pos_proba
        for token in get_ngrams(text, (2, 2)):
            if token in trained_data:
                pred_pos = pred_pos - \
                    trained_data[token][1] + trained_data[token][0]
        pred_pos += pos_log_prior

        pred_neg = prior_neg_proba
        for token in get_ngrams(text, (2, 2)):
            if token in trained_data:
                pred_neg = pred_neg - \
                    trained_data[token][3] + trained_data[token][2]
        pred_neg += neg_log_prior

        if pred_pos > pred_neg:
            preds.append('neg')
        else:
            preds.append('pos')
    return preds
