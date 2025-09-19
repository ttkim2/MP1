# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.5, bigram_laplace=0.5, bigram_lambda=0.3, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    #Count Unigrams and Bigrams
    def count_grams(train_set, train_labels):
        positive_unigrams = Counter()
        negative_unigrams = Counter()
        positive_bigrams = Counter()
        negative_bigrams = Counter()
        positive_total_unigrams = 0
        negative_total_unigrams = 0
        positive_total_bigrams = 0
        negative_total_bigrams   = 0
        for tokens, label in zip(train_set, train_labels):
            if label == 1:
                positive_unigrams.update(tokens)
                positive_total_unigrams += len(tokens)
            else:
                negative_unigrams.update(tokens)
                negative_total_unigrams += len(tokens)
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])
                if label == 1:
                    positive_bigrams[bigram] += 1
                    positive_total_bigrams += 1
                else:
                    negative_bigrams[bigram] += 1
                    negative_total_bigrams += 1
        return positive_unigrams, negative_unigrams, positive_bigrams, negative_bigrams, positive_total_unigrams, negative_total_unigrams, positive_total_bigrams, negative_total_bigrams
    positive_unigrams, negative_unigrams, positive_bigrams, negative_bigrams, positive_total_unigrams, negative_total_unigrams, positive_total_bigrams, negative_total_bigrams = count_grams(train_set, train_labels)
    unigram_vocab = set(positive_unigrams.keys()) | set(negative_unigrams.keys())
    bigram_vocab  = set(positive_bigrams.keys()) | set(negative_bigrams.keys())
    Size_unigrams = len(unigram_vocab)
    Size_bigrams  = len(bigram_vocab)

    #Log Probability
    def log_prob_unigram(tokens, target_class):
        logp = 0.0
        for i in tokens:
            if target_class == 1:
                count = positive_unigrams[i]
                num = count + unigram_laplace
                denom = positive_total_unigrams + unigram_laplace * Size_unigrams
            else:
                count = negative_unigrams[i]
                num = count + unigram_laplace
                denom = negative_total_unigrams + unigram_laplace * Size_unigrams
            logp += math.log(num / denom)
        return logp
    def log_prob_bigram(tokens, target_class):
        logp = 0.0
        for i in range(len(tokens) - 1):
            j = (tokens[i], tokens[i+1])
            if target_class == 1:
                count = positive_bigrams[j]
                num = count + bigram_laplace
                denom = positive_total_bigrams + bigram_laplace * Size_bigrams
            else:
                count = negative_bigrams[j]
                num = count + bigram_laplace
                denom = negative_total_bigrams + bigram_laplace * Size_bigrams
            logp += math.log(num / denom)
        return logp
    #Score
    yhats = []
    for tokens in tqdm(dev_set, disable=silently):
        positive_score = math.log(pos_prior)
        positive_score += (1 - bigram_lambda) * log_prob_unigram(tokens, 1)
        positive_score += bigram_lambda * log_prob_bigram(tokens, 1)
        negative_score = math.log(1 - pos_prior)
        negative_score += (1 - bigram_lambda) * log_prob_unigram(tokens, 0)
        negative_score += bigram_lambda * log_prob_bigram(tokens, 0)
        if positive_score > negative_score:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats





