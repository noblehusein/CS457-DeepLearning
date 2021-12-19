from keras.datasets import mnist
from matplotlib import pyplot

from matplotlib.pyplot import figure

from keras.datasets import mnist
from matplotlib import pyplot
import csv
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.optimizers import SGD

def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	#trainX=trainX[1:10000]
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	#trainY=trainY[1:10000]
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY


	# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

	# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_uniform', input_shape=(28, 28, 1), padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_uniform',padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((3, 3)))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


	# evaluate a model using k-fold cross-validation
def evaluate_model(trainX, trainY, testX, testY):
	scores, histories = list(), list()
	# prepare cross validation
	
	# enumerate splits
	
	# define model
	model = define_model()
	# select rows for train and test
	#trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
	# fit model
	history = model.fit(trainX, trainY, epochs=100, batch_size=256, validation_data=(testX, testY))
	# evaluate model
	_, acc = model.evaluate(testX, testY)
	print('> %.3f' % (acc * 100.0))
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
	plt.show()
	return history


	# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	history=evaluate_model(trainX, trainY, testX, testY)
	return history
history=run_test_harness()

model = define_model()
model.summary()