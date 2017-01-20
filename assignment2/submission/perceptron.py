# perceptron.py
# -------------
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


# Perceptron implementation
import util
import numpy as np
import sys
import random
PRINT = True

np.random.seed(42)
class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;

    def train( self, trainingData, trainingLabels, validationData, validationLabels, validate):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the assignment description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector of values).
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        f = open("perceptronIterations.csv","w")
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                if(validate):
                    if (i % (len(trainingData)/2) ==0):
                        guesses = self.classify(validationData)
                        correct = [guesses[j] == validationLabels[j] for j in range(len(validationLabels))].count(True)
                        f.write(str(i + iteration*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n")
                "*** YOUR CODE HERE ***"
                label = 0
                max_score = 0
                for j in self.legalLabels:
                    curr_score = self.weights[j]*trainingData[i]
                    if(max_score<curr_score):
                        max_score = curr_score
                        label = j
                if(label!=trainingLabels[i]):
                    self.weights[label] -= trainingData[i]
                    self.weights[trainingLabels[i]] += trainingData[i]

                "util.raiseNotDefined()"
                
        if(validate):
            guesses = self.classify(validationData)
            correct = [guesses[j] == validationLabels[j] for j in range(len(validationLabels))].count(True)
            f.write(str(i + iteration*len(trainingData))+","+str(100*correct/(1.0*len(trainingData)))+"\n")                        
                
        f.close()

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the assignment description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
