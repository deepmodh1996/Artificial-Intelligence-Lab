from math import *
import random
from copy import deepcopy

def argmin(values):
    return min(enumerate(values), key=lambda x: x[1])[0]
def avg(values):
    return float(sum(values))/len(values)

def readfile(filename):
    '''
    File format: Each line contains a comma separated list of real numbers, representing a single point.
    Returns a list of N points, where each point is a d-tuple.
    '''
    data = []
    with open(filename, 'r') as f:
        data = f.readlines()
    data = [tuple(map(float, line.split(','))) for line in data]
    return data

def writefile(filename, means):
    '''
    means: list of tuples
    Writes the means, one per line, into the file.
    '''
    if filename == None: return
    with open(filename, 'w') as f:
        for m in means:
            f.write(','.join(map(str, m)) + '\n')
    print 'Written means to file ' + filename


def distance_euclidean(p1, p2):
    '''
    p1: tuple: 1st point
    p2: tuple: 2nd point

    Returns the Euclidean distance b/w the two points.
    '''
    ssum = 0.0
    # print len(p1)
    # print p1
    for i in range(0,len(p1)):
    	ssum += ( p1[i]-p2[i] ) * ( p1[i]-p2[i] )
    distance = sqrt(ssum)
    

    # TODO [task1]:
    # Your function must work for all sized tuples.

    ########################################
    return distance

def distance_manhattan(p1, p2):
    '''
    p1: tuple: 1st point
    p2: tuple: 2nd point

    Returns the Manhattan distance b/w the two points.
    '''

    # k-means uses the Euclidean distance.
    # Changing the distant metric leads to variants which can be more/less robust to outliers,
    # and have different cluster densities. Doing this however, can sometimes lead to divergence!

    distance = 0.0
    for i in range(0,len(p1)):
    	distance += abs( p1[i]-p2[i] )
   

    # TODO [task1]:
    # Your function must work for all sized tuples.

    ########################################
    return distance

def initialization_forgy(data, k):
    '''
    data: list of tuples: the list of data points
    k: int: the number of cluster means to return

    Returns a list of tuples, representing the cluster means 
    '''
    dataTemp = data
    random.shuffle(dataTemp)
    means = dataTemp[0:k]

    # TODO [task1]:
    # Use the Forgy algorithm to initialize k cluster means.

    ########################################
    assert len(means) == k
    return means


def initialization_randompartition(data, distance, k):
    '''
    data: list of tuples: the list of data points
    distance: callable: a function implementing the distance metric to use
    k: int: the number of cluster means to return

    Returns a list of tuples, representing the cluster means 
    '''
    means = []

    dimension = len(data[0])
    #clusterSum[i] contains sum of datapoints in ith cluster
    #numPoint[i] contains number of datapoints in ith cluster
    #initialising clusterSum[i] by x0 : which is origin in dimenstion of datapoint
    #initialising numberPoint[i] to 0
    x0 = []
    for i in range(0,dimension):
    	x0.append(0)
    clusterSum = [tuple(x0)]*k
    numPoint = [0]*k
    # iterating over each datapoint
    for i in range(0,len(data)):
    	c = random.randint(0,k-1) # randomly assiging cluster to ith data point
    	# maintaining clusterSum array
    	if(clusterSum[c]==0):
    		clusterSum[c] = data[i]
    	else:
    		x = []
    		for r in range(0,dimension):
    			x.append(clusterSum[c][r] + data[i][r])
    		clusterSum[c] = tuple(x)
    	numPoint[c] += 1 # maintaining numPoint array

    # calculating mean of each cluster = clusterSum[i]/numPoint[i]
    for i in range(0,k):
    	z = clusterSum[i]
    	# since we have number of data points sufficiently large, and we randomly assign cluser;
    	# numPoint in any cluster will very highly likely to be nonzero
    	# however if in very unlikely case when numPoint is zero (which will almost never happen);
    	# we take it's mean to be origin in dimenstion of datapoints ( that is x0 see line 111-113)
    	if (numPoint[i] != 0):
    		x = []
	    	for j in range(0,dimension):
	    		x.append(z[j]/(1.0*numPoint[i]))
    		y = tuple(x)
    		means.append(y)
    	else:
    		means.append(x0)

    # TODO [task3]:
    # Use the randompartition algorithm to initialize k cluster means.
    # Make sure you use the distance function given as parameter.

    # NOTE: Provide extensive comments with your code.

    ########################################
    assert len(means) == k
    return means

def initialization_kmeansplusplus(data, distance, k):
    '''
    data: list of tuples: the list of data points
    distance: callable: a function implementing the distance metric to use
    k: int: the number of cluster means to return

    Returns a list of tuples, representing the cluster means 
    '''

    means = []
    means.append(random.choice(data)) # randomly choosing 1st center from data points
    for r in range(1,k):
    	D = [] # D[i] contains square of distance of ith datapoint to nearest cluster mean already chosen
    	sumD = 0.0 # contains D[0] + D[1] + ... + D[i] : i index seen so far in following loop
    	# NOTE : we don't need to rule out points explicitely which are not cluster centers (in formula of weight counting)
    	#        because it's D value is 0, and hence will not contribute in any way.
    	for i in range(0,len(data)):
    		# calculating minimum distance between ith data point and nearest cluster mean so far
    		minDist = distance(means[0], data[i])
    		for j in range(0,len(means)):
    			distTemp = distance(means[j], data[i])
    			if(distTemp < minDist):
    				minDist = distTemp
    		D.append(minDist*minDist) # appending minimum distance square to D
    		sumD += (minDist*minDist) # maintaining sumD
    	# dividing each element of D by sumD
    	D[:] = [x / sumD for x in D]
    	# hence from now on D[i] is the probability of picking ith point as new mean

    	# to pick next mean with probability distribution stored in D; concept of cdf is used
    	# going from 0th datapoint to ith data point, cdf increases.
    	# at each datapoint cdf increases proportional to probability of that datapoint.
    	# hence if we randomly choose a number say x between 0 to 1 and associate it with cdf and pick the 1st index after which cdf is < x
    	# then since increment of cdf is proportional to probability; we end up picking point based on weights as mentioned in algorithm
    	x = random.random() # x is any random number between 0 and 1
    	index = -1
    	cdf = 0.0
    	for i in range(0,len(data)):
    		cdf += D[i]
    		if (cdf >= x):
    			index = i
    			break
    	# if index is -1 that is x corrosponds to cdf of element in last index
    	if (index == -1):
    		index = len(data) - 1
    	means.append(data[index])

    
    # TODO [task3]:
    # Use the kmeans++ algorithm to initialize k cluster means.
    # Make sure you use the distance function given as parameter.

    # NOTE: Provide extensive comments with your code.

    ########################################
    assert len(means) == k
    return means


def iteration_one(data, means, distance):
    '''
    data: list of tuples: the list of data points
    means: list of tuples: the current cluster centers
    distance: callable: function implementing the distance metric to use

    Returns a list of tuples, representing the new cluster means after 1 iteration of k-means clustering algorithm.
    '''

    new_means = []
    k = len(means)
    dimension = len(data[0])

    x0 = []
    for i in range(0,dimension):
    	x0.append(0)
    clusterSum = [tuple(x0)]*k # sum of all the tuples in cluster
    numPoint = [0]*k # number of points in cluster
    for i in range(0,len(data)):
    	c = 0
    	minDist = distance(means[0], data[i])
    	for j in range(0,k):
    		distTemp = distance(means[j], data[i])
    		if(distTemp < minDist):
    			c = j
    			minDist = distTemp
    	if(clusterSum[c]==0):
    		clusterSum[c] = data[i]
    	else:
    		x = []
    		for r in range(0,dimension):
    			x.append(clusterSum[c][r] + data[i][r])
    		clusterSum[c] = tuple(x)
    	numPoint[c] += 1

    for i in range(0,k):
    	z = clusterSum[i]
    	if (numPoint[i] != 0):
    		x = []
	    	for j in range(0,dimension):
	    		x.append(z[j]/(1.0*numPoint[i]))
    		y = tuple(x)
    		new_means.append(y)
    	else:
    		new_means.append(means[i])

    # TODO [task1]:
    # You must find the new cluster means.
    # Perform just 1 iteration (assignment+updation)

    ########################################
    return new_means

def hasconverged(old_means, new_means, epsilon=1e-1):
    '''
    old_means: list of tuples: The cluster means found by the previous iteration
    new_means: list of tuples: The cluster means found by the current iteration

    Returns true iff no cluster center moved more than epsilon distance.
    '''

    converged = True
    for i in range(0,len(old_means)):
    	if(distance_euclidean(old_means[i],new_means[i]) > epsilon):
    		converged = False
    		break
    # TODO [task1]:
    # Use Euclidean distance to measure centroid displacements.

    ########################################
    return converged



def iteration_many(data, means, distance, maxiter, epsilon=1e-1):
    '''
    maxiter: int: Number of iterations to perform

    Uses the iteration_one function.
    Performs maxiter iterations of the k-means clustering algorithm, and saves the cluster means of all iterations.
    Stops if convergence is reached earlier.

    Returns:
    all_means: list of (list of tuples): Each element of all_means is a list of the cluster means found by that iteration.
    '''
    
    all_means = []
    all_means.append(means)

    for i in range(0,maxiter):
    	means2 = (iteration_one(data, means, distance))
    	all_means.append(means2)
    	means = means2
    	if(hasconverged(all_means[-2],all_means[-1])):
    		break

    # TODO [task1]:
    # Make sure you've implemented the iteration_one, hasconverged functions.
    # Perform iterations by calling the iteration_one function multiple times.
    # Stop only if convergence is reached, or if max iterations have been exhausted.
    # Save the results of each iteration in all_means.
    # Tip: use deepcopy() if you run into weirdness.

    ########################################

    return all_means



def performance_SSE(data, means, distance):

    '''
    data: list of tuples: the list of data points
    means: list of tuples: representing the cluster means 

    Returns: The Sum Squared Error of the clustering represented by means, on the data.
    '''

    sse = 0.0
    for i in range(0,len(data)):
    	c = 0
    	minDist = distance(means[0], data[i])
    	for j in range(0,len(means)):
    		distTemp = distance(means[j], data[i])
    		if(distTemp <= minDist):
    			c = j
    			minDist = distTemp
    	d = distance_euclidean(means[c],data[i])
    	sse += (d*d)

    # TODO [task1]:
    # Calculate the Sum Squared Error of the clustering represented by means, on the data.
    # Make sure to use the distance metric provided.

    ########################################
    return sse



########################################################################
##                      DO NOT EDIT THE FOLLWOING                     ##
########################################################################


import sys
import argparse
import matplotlib.pyplot as plt
from itertools import cycle
from pprint import pprint as pprint

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', type=str, help='Required. Dataset filename')
    parser.add_argument('-o', '--output', dest='output', type=str, help='Output filename')
    parser.add_argument('-iter', '--iter', '--maxiter', dest='maxiter', type=int, default=10000, help='Maximum number of iterations of the k-means algorithm to perform. (may stop earlier if convergence is achieved)')
    parser.add_argument('-e', '--eps', '--epsilon', dest='epsilon', type=float, default=1e-1, help='Minimum distance the cluster centroids move b/w two consecutive iterations for the algorithm to continue.')
    parser.add_argument('-init', '--init', '--initialization', dest='init', type=str, default='forgy', help='The initialization algorithm to be used. {forgy, randompartition, kmeans++}')
    parser.add_argument('-dist', '--dist', '--distance', dest='dist', type=str, default='euclidean', help='The distance metric to be used. {euclidean, manhattan}')
    parser.add_argument('-k', '--k', dest='k', type=int, default=5, help='The number of clusters to use.')
    parser.add_argument('-verbose', '--verbose', dest='verbose', type=bool, default=False, help='Turn on/off verbose.')
    parser.add_argument('-seed', '--seed', dest='seed', type=int, default=0, help='The RNG seed.')
    parser.add_argument('-numexperiments', '--numexperiments', dest='numexperiments', type=int, default=1, help='The number of experiments to run.')
    _a = parser.parse_args()

    if _a.input is None:
        print 'Input filename required.\n'
        parser.print_help()
        sys.exit(1)
    
    args = {}
    for a in vars(_a):
        args[a] = getattr(_a, a)

    if _a.init.lower() in ['random', 'randompartition']:
        args['init'] = initialization_randompartition
    elif _a.init.lower() in ['k++', 'kplusplus', 'kmeans++', 'kmeans', 'kmeansplusplus']:
        args['init'] = initialization_kmeansplusplus
    elif _a.init.lower() in ['forgy', 'frogy']:
        args['init'] = initialization_forgy
    else:
        print 'Unavailable initialization function.\n'
        parser.print_help()
        sys.exit(1)


    if _a.dist.lower() in ['manhattan', 'l1', 'median']:
        args['dist'] = distance_manhattan
    elif _a.dist.lower() in ['euclidean', 'euclid', 'l2']:
        args['dist'] = distance_euclidean
    else:
        print 'Unavailable distance metric.\n'
        parser.print_help()
        sys.exit(1)

    print '-'*40 + '\n'
    print 'Arguments:'
    pprint(args)
    print '-'*40 + '\n'
    return args

def visualize_data(data, all_means, args):
    print 'Visualizing...' 
    means = all_means[-1]
    k = args['k']
    distance = args['dist']
    clusters = [[] for _ in range(k)]
    for point in data:
        dlist = [distance(point, center) for center in means]
        clusters[argmin(dlist)].append(point)

    # plot each point of each cluster
    colors = cycle('rgbwkcmy')

    for c, points in zip(colors, clusters):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.scatter(x,y, c = c)

    # plot each cluster centroid
    colors = cycle('krrkgkgr')
    colors = cycle('rgbkkcmy')

    for c, clusterindex in zip(colors, range(k)):
        x = [iteration[clusterindex][0] for iteration in all_means]
        y = [iteration[clusterindex][1] for iteration in all_means]
        plt.plot(x,y, '-x', c = c, linewidth='1', mew=15, ms=2)
    plt.axis('equal')
    plt.show()

def visualize_performance(data, all_means, distance):

    errors = [performance_SSE(data, means, distance) for means in all_means]
    plt.plot(range(len(all_means)), errors)
    plt.title('Performance plot')
    plt.xlabel('Iteration')
    plt.ylabel('Sum Squared Error')
    plt.show()


if __name__ == '__main__':

    args = parse()
    # Read data
    data = readfile(args['input'])
    print 'Number of points in input data: {}\n'.format(len(data))
    verbose = args['verbose']

    totalSSE = 0
    totaliter = 0

    for experiment in range(args['numexperiments']):
        print 'Experiment: {}'.format(experiment+1)
        random.seed(args['seed'] + experiment)
        print 'Seed: {}'.format(args['seed'] + experiment)

        # Initialize means
        means = []
        if args['init'] == initialization_forgy:
            means = args['init'](data, args['k']) # Forgy doesn't need distance metric
        else:
            means = args['init'](data, args['dist'], args['k'])

        if verbose:
            print 'Means initialized to:'
            print means
            print ''

        # Run k-means clustering
        all_means = iteration_many(data, means, args['dist'], args['maxiter'], args['epsilon'])

        SSE = performance_SSE(data, all_means[-1], args['dist'])
        totalSSE += SSE
        totaliter += len(all_means)-1
        print 'Sum Squared Error: {}'.format(SSE)
        print 'Number of iterations till termination: {}'.format(len(all_means)-1)
        print 'Convergence achieved: {}'.format(hasconverged(all_means[-1], all_means[-2]))


        if verbose:
            print '\nFinal means:'
            print all_means[-1] 
            print ''
            
    print '\n\nAverage SSE: {}'.format(float(totalSSE)/args['numexperiments'])
    print 'Average number of iterations: {}'.format(float(totaliter)/args['numexperiments'])

    if args['numexperiments'] == 1:
        # save the result
        writefile(args['output'], all_means[-1])

        # If the data is 2-d and small, visualize it.
        if len(data) < 5000 and len(data[0]) == 2:
            visualize_data(data, all_means, args)

        visualize_performance(data, all_means, args['dist'])
