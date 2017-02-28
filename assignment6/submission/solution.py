import sys

filename = sys.argv[1]
# print filename
f = open(filename,'r')
MDP = f.read().replace('\n','\t').split('\t')
MDP = [x for x in MDP if x]
# print MDP

S = int(MDP[0])
A = int(MDP[1])
# print S
# print A

r = 2
R = [[[0 for x in range(S)] for y in range(A)] for z in range(S)]
for i in range(0, S):
	for j in range(0, A):
		for k in range(0, S):
			R[i][j][k] = float(MDP[r])
			r = r + 1
# print R

T = [[[0 for x in range(S)] for y in range(A)] for z in range(S)]
for i in range(0, S):
	for j in range(0, A):
		for k in range(0, S):
			T[i][j][k] = float(MDP[r])
			r = r + 1
# print T

gamma = float(MDP[r])
# print gamma

V = [0]*S
PI = [0]*S
t = 0
epsilon = 10**(-16)
while (True):
	t = t + 1
	Vprev = V[:]
	# print Vprev
	for i in range(0, S):
		maxvalue = 0.0
		PI[i] = 0
		for k in range(0, S):
			maxvalue += T[i][0][k]*(R[i][0][k] + gamma*Vprev[k])
		for j in range(0, A):
			value = 0.0
			for k in range(0, S):
				value += T[i][j][k]*(R[i][j][k] + gamma*Vprev[k])
			if (value > maxvalue):
				maxvalue = max(value, maxvalue)
				PI[i] = j
		V[i] = maxvalue

	numValid = 0
	for i in range(0, len(V)):
		if(abs(V[i] - Vprev[i]) > epsilon):
			numValid = 1
	if (numValid == 0):
		break

for i in range(0, S):
	print str(V[i]) + '\t' + str(PI[i])
print "Iterations" + '\t' + str(t)
