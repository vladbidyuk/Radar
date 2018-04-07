from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import tflearn

PLATE_MODEL_NAME = 'cnnModels/Plates/plates-0.001-6conv-basic.model'
CHARS_MODEL_NAME = 'cnnModels/Characters/characters-0.001-4conv-basic.model'

PLATE_IMG_SIZE, CHARS_IMG_SIZE = 200, 50
LR = 1e-3   #0.001

#~~~Plate model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tf.reset_default_graph()
convnet = input_data(shape=[None, PLATE_IMG_SIZE, PLATE_IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu');  convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu');    convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

plate_model = tflearn.DNN(convnet, tensorboard_dir='log')
plate_model.load(PLATE_MODEL_NAME)
#~~~Plate model~~~~END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~Characters model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tf.reset_default_graph()
convnet = input_data(shape=[None, CHARS_IMG_SIZE, CHARS_IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu');   convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu');    convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 37, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

chars_model = tflearn.DNN(convnet, tensorboard_dir='log')
chars_model.load(CHARS_MODEL_NAME)
#~~~Characters model~~~~END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
