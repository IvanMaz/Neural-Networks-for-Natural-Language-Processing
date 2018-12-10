from os import listdir
from math import log
import string
import re
from statistics import mean, median
from collections import Counter, defaultdict


punct = string.punctuation + '…–—‘“‚„«»'

STOPWORDS = {'now', 'up', 'against', 'again', 'from', 'or', 'did', 'here', 'his', 'those', 'had',
             'few', 'own', 'ma', 'all', 'where', 's', 'are', 'being', 'herself', 'ourselves',
             'themselves', 'and', 'at', 'can', 'i', 'don', 'why', 'ours', 'they', 'wasn', 'be', 'hasn',
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

    count_mode = 'mul'
    ngram_range = (1, 4)
    min_cnt = 1
    max_df = 0.7

    train_pos_total, train_neg_total = len(train_pos_tok), len(train_neg_tok)
    dict_total = train_pos_total + train_neg_total

    mul_pos_idx, mul_neg_idx = count(
        train_pos_tok, train_neg_tok, count_mode, ngram_range)

    if min_cnt or max_df < 1:
        tmp_pos_idx = defaultdict(lambda: 1)
        for k, v in mul_pos_idx.items():
            if v > min_cnt+1 and v < max_df*train_pos_total:
                tmp_pos_idx[k] = v
        tmp_neg_idx = defaultdict(lambda: 1)
        for k, v in mul_neg_idx.items():
            if v > min_cnt+1 and v < max_df*train_neg_total:
                tmp_neg_idx[k] = v
        mul_pos_idx = tmp_pos_idx
        mul_neg_idx = tmp_neg_idx

    unique_words = sorted(
        list(set(mul_pos_idx.keys()) | set(mul_neg_idx.keys())))

    train_pos_feat_total = sum([mul_pos_idx[k] for k in unique_words])
    train_neg_feat_total = sum([mul_neg_idx[k] for k in unique_words])

    trained_data = {k: [log(mul_pos_idx[k]) - log(train_pos_feat_total),
                        log(mul_neg_idx[k]) - log(train_neg_feat_total)] for k in unique_words}

    pos_log_prior = log(train_pos_total / dict_total)
    neg_log_prior = log(train_neg_total / dict_total)

    return pos_log_prior, neg_log_prior, ngram_range, trained_data


def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """

    pos_log_prior, neg_log_prior, ngram_range, trained_data = params

    texts_tok = [tokenize(r) for r in texts]

    preds = []
    for text in texts_tok:
        pred_pos = 0
        for word in get_ngrams(text, ngram_range):
            if word in trained_data:
                pred_pos += trained_data[word][0]
        pred_pos += pos_log_prior

        pred_neg = 0
        for word in get_ngrams(text, ngram_range):
            if word in trained_data:
                pred_neg += trained_data[word][1]
        pred_neg += neg_log_prior

        if pred_pos > pred_neg:
            preds.append('neg')
        else:
            preds.append('pos')
    return preds
