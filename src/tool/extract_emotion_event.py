import json
import os
import random
import string

import nltk.sentiment.sentiment_analyzer
import numpy as np
import torch
from bert_score import BERTScorer
from nltk import sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from multiprocessing import Pool


random.seed(2020)
stop = stopwords.words('english') + list(string.punctuation)  # + ["'s", "'m", "'re", "'ve"]

sid = SentimentIntensityAnalyzer()
stemmer = WordNetLemmatizer()

discard_words = []

def extract_emotion_event_at_least_one(text, limit=False, per_sent=False):
    def get_wordnet_pos(tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatize(word, nltk_tag):
        tag = get_wordnet_pos(nltk_tag)
        return stemmer.lemmatize(word, tag).lower()

    def sample_sorted(words, num):
        ids = list(range(len(words)))
        choosed = random.sample(ids, num)
        choosed = sorted(choosed)
        res = []
        for i in choosed:
            res.append(words[i])
        return res

    def extract_emotion_event(sent, last_word):
        global sid, discard_words

        words = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(words)
        res = []
        res.append(last_word)
            

        for word, tag in tagged:
            if not word[0].isalpha():
                continue
            origin_word = lemmatize(word, tag)

            ss = sid.polarity_scores(word)
            for key in ss:
                if not word[0].isalpha():
                    continue
                if ss[key] > 0.5 and (key == 'pos' or key == 'neg'):
                   
                    res.append(origin_word)

            if get_wordnet_pos(tag) in [wordnet.VERB, wordnet.NOUN]:
                if origin_word in stop or word in stop:
                    continue
                if origin_word in discard_words or word in discard_words:
                    continue
                res.append(origin_word)

        res.pop(0)
        from math import ceil
        max_len = min(5, ceil(len(words) * 0.1))
        if limit and max_len < len(res):
            return sample_sorted(res, max_len)
        else:
            return res

    def choose_one_word(sent):
        words = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(words)
        new_tagged = [(word, tag) for word, tag in tagged if word[0].isalpha()]
        if not new_tagged:
            word, tag = random.choice(tagged)
        else:
            word, tag = random.choice(new_tagged)
        return [lemmatize(word, tag)]

    if isinstance(text, list):
        sents = text
    else:
        sents = nltk.sent_tokenize(text)

    res = []
    choose_one_word_cnt = 0
    for sent in sents:
        last_word = None
        if res:
            last_word = res[-1][-1]
        w = extract_emotion_event(sent, last_word)
        if not per_sent:
            res.extend(w)
        else:
            if w:
                res.append(w)
            else:
                choose_one_word_cnt += 1
                res.append(choose_one_word(sent))

    if per_sent:
        return res, choose_one_word_cnt

    if res:
        return res, 0
    else:
        return choose_one_word(sents[0]), 1

