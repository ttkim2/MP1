# naive_bayes.py
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
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
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
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)
    
    count_positive = Counter()
    count_negative = Counter()

    for review, label in zip(train_set, train_labels):
        if label == 1:
            count_positive.update(review)
        else:
            count_negative.update(review)
    total_count_positive = sum(count_positive.values())
    total_count_negative = sum(count_negative.values())
    vocabulary_positive = set(count_positive.keys())
    vocabulary_negative = set(count_negative.keys())
    vocabulary_combined = vocabulary_positive.union(vocabulary_negative)
    total_unique_words = len(vocabulary_combined)
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        positive_log = math.log(pos_prior)
        negative_log = math.log(1 - pos_prior)
        for word in doc:
            positive_log += math.log(
                (count_positive.get(word, 0) + laplace) / (total_count_positive + laplace*total_unique_words)
            )
            negative_log += math.log(
                (count_negative.get(word, 0) + laplace) / (total_count_negative + laplace*total_unique_words)
            )
        if (positive_log > negative_log):
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats
