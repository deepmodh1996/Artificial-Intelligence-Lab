# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
import numpy as np
import sys
import random
PRINT = True

np.random.seed(42)


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels, validate):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008, 0.016, 0.032]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        
        f = open("perceptronIterations.csv","w")
        W = []
        for C in Cgrid:
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "... for C = ", C
                for i in range(len(trainingData)):
                    data = trainingData[i].copy()
                    label = 0
                    max_score = 0
                    for j in self.legalLabels:
                        curr_score = self.weights[j]*data
                        if(max_score<curr_score):
                            max_score = curr_score
                            label = j
                    if(label!=trainingLabels[i]):
                        a = (self.weights[label] - self.weights[trainingLabels[i]])*data
                        t = min(C,(a+1.0)/(2.0*(data*data)))
                        data.mulAll(t)
                        self.weights[label] -= data
                        self.weights[trainingLabels[i]] += data
            W.append(self.weights)
            self.weights = {}
            for label in self.legalLabels:
                self.weights[label] = util.Counter() 

        index = 0
        max_accuracy = 0
        C_max_index = 0
        for C in Cgrid:
            self.weights = W[index]
            index = index + 1
            guesses = self.classify(validationData)
            correct = [guesses[j] == validationLabels[j] for j in range(len(validationLabels))].count(True)
            accuracy = 100*correct/(1.0*len(trainingData))
            if(max_accuracy<accuracy):
                max_accuracy = accuracy
                C_max_index = index
        self.weights = W[C_max_index]

        f.close()
        "util.raiseNotDefined()"        

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


