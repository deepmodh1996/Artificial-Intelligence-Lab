import numpy as np
# Use command line arguements for Task 1 (sys.argv)
import sys

# the answer to life, universe and everything. also for reproducibility
np.random.seed(42)  

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train_1, train_labels_1), (X_test, test_labels_1) = mnist.load_data()
 
# Preprocess input data
X_train_1 = X_train_1.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train_1 = X_train_1.astype('float32')
X_test = X_test.astype('float32')

X_train_1 /= 255
X_test /= 255

# Divides the dataset into train and validation sets
X_valid = X_train_1[50000:60000]
X_train = X_train_1[:1000]
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'validation samples')
 
# Preprocess class labels
train_labels = np_utils.to_categorical(train_labels_1, 10)
test_labels = np_utils.to_categorical(test_labels_1, 10)
valid_labels = train_labels[50000:60000]
train_labels = train_labels[:1000]

# Define model architecture
model = Sequential()

model.add(Dense( 100, input_shape=(784,)))
model.add(Activation('relu'))

for i in range(5):
	model.add(Dense(100))
	model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))



# model.add(Dense(100))
# model.add(Activation('relu'))

# model.add(Dense(10))
# model.add(Activation('softmax'))

# Compile model
sgd = SGD(lr=0.02) # Sets learning rate. 
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
 
# Fit model on training data
model.fit(X_train, train_labels, 
          batch_size=16, nb_epoch=500, verbose=1)
 
# Evaluate model on test data.
score = model.evaluate(X_valid, valid_labels, verbose=0)
score_test = model.evaluate(X_test, test_labels, verbose=0)
# This returns only a score, so you will need to use another function for 
# extracting predicted labels for your confusion matrix. Use this line for that:
classes = model.predict_classes(X_valid, batch_size=16)

print('Validation score:', score[0])
print('Validation accuracy:', score[1])
print('Test score:', score_test[0])
print('Test accuracy:', score_test[1])
f=open("confusion.txt","a")
# f.write(sys.argv[1]+","+str(score[1])+"\n" )
M = np.zeros(shape = (10,10))
# print valid_labels[0].argmax()
# print valid_labels[1].argmax()

for i in range(len(classes)):
	k1 = valid_labels[i].argmax()
	k2 = int(classes[i])
	M[k1][k2] = M[k1][k2] + 1
	# M[int(valid_labels[i].argmax())][int(classes[i])] = M[int(valid_labels[i].argmax())][int(classes[i])] + 1

for i in range(10):
	for j in range(10):
		f.write(str(M[i][j])+" ")
	f.write("\n")
f.close()
