#! /usr/bin/python

import random

S = 20
A = 10
gamma = 0.99

randomSeed = 0

random.seed(randomSeed)

T = []
R = []
for s in range(0, S):

    T.append([])
    R.append([])

    for a in range(0, A):
        T[s].append([])
        R[s].append([])

        sum = 0

        for sPrime in range(0, S):
            T[s][a].append(random.random());
            sum = sum + T[s][a][sPrime]
            R[s][a].append(random.random());
            
        for sPrime in range(0, S):
            T[s][a][sPrime] /= sum


print S
print A

for s in range(0, S):
    for a in range(0, A):
        for sPrime in range(0, S):
            print str(R[s][a][sPrime]) + "\t",

        print "\n",

for s in range(0, S):
    for a in range(0, A):
        for sPrime in range(0, S):
            print str(T[s][a][sPrime]) + "\t",

        print "\n",

print gamma

