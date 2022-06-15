# -*- coding: utf-8 -*-
"""Unet model"""

# standard library

# external

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv

from tensorflow.python.keras.models import Sequential, model_from_json, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten

from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
# from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import TimeDistributed, Conv1D, MaxPooling1D

from dataloader.dataloader import DataLoader
from utils.logger import get_logger
from sklearn.metrics import classification_report

# internal
from .base_model import BaseModel

LOG = get_logger('mHealth')


class MHet(BaseModel):
    """Mhealth Model Class"""

    def __init__(self, config):
        super().__init__(config)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.model.input, include_top=False)
        self.model = None
        self.output_channels = self.config.model.output

        self.dataset = None
        self.info = None
        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches
        self.val_subsplits = self.config.train.val_subsplits
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0
        self.url = self.config.mhealth.url
        self.seq_size = self.config.mhealth.mblock_size
        self.subject = self.config.mhealth.subject_count

        self.image_size = self.config.data.image_size
        self.xtrain_dataset = []
        self.xtest_dataset = []
        self.xvalid_dataset = []
        self.yvalid_dataset = []
        self.ytrain_dataset = []
        self.ytest_dataset = []
        self.read_dataX = []
        self.read_dataY = []

        self.xtrain_valid_dataset = []
        self.ytrain_valid_dataset = []

        self.y_train = []
        self.y_valid = []
        self.y_test = []

    def load_data(self):
        """Downloads, Loads and Preprocess data """
        # LOG.info(f'Downloading {self.url} dataset.....')

        # DataLoader.download_and_extract(self.config.mhealth)

        LOG.info(f'Collecting and saving at {self.config.data.path} path in the current directory...')

        DataLoader.collect_save_data(self.subject, self.seq_size)

        LOG.info(f'Read the data set for dataX and dataY.... from {self.config.data.path}')

        self.read_dataX, self.read_dataY = DataLoader.load_data(self.config.mhealth.path_readX,
                                                                self.config.mhealth.path_readY)

        LOG.info(f' split into a training/test')

        self.xtrain_dataset, self.xtest_dataset, \
        self.ytrain_dataset, self.ytest_dataset = DataLoader.preprocess_train_test(self.read_dataX, self.read_dataY)

        LOG.info(f' standardize the training data set')

        self.xtrain_dataset, self.xtest_dataset = DataLoader.preprocess_standardize_train_dataset(self.xtrain_dataset,
                                                                                                  self.xtest_dataset)

        LOG.info(f' train and validation')

        self.xtrain_valid_dataset, self.xvalid_dataset, \
        self.ytrain_valid_dataset, self.yvalid_dataset = DataLoader.preprocess_train_validation_dataset(
            self.xtrain_dataset,
            self.ytrain_dataset)

        LOG.info(f' one-hot encode')

        self.y_train = DataLoader.preprocess_onhot_encoder(self.ytrain_valid_dataset)
        self.y_valid = DataLoader.preprocess_onhot_encoder(self.yvalid_dataset)
        self.y_test = DataLoader.preprocess_onhot_encoder(self.ytest_dataset)


    def evaluate_model(self):

        """ Evaluate model"""

        # Hyperparameters
        #batch_size = 64
        seq_len = 100
        learning_rate = 0.0001
        #epochs = 15

        n_classes = 12
        n_channels = 23

        ## here is where I start make some modifications to integrates the keras framework

        verbose = 0

        n_timesteps, n_features, n_outputs = self.xtrain_valid_dataset.shape[1], self.xtrain_valid_dataset.shape[2], \
                                             self.y_train.shape[1]

        # define model

        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(n_outputs, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit network

        self.model.fit(self.xtrain_valid_dataset, self.y_train, epochs=self.epoches, batch_size=self.batch_size, verbose=0)

        # evaluate model

        _, accuracy = self.model.evaluate(self.xtest_dataset, self.y_test, batch_size=self.batch_size, verbose=0)

        return accuracy


    def build(self):
        """ Builds the Keras model based """

        scores = list()
        repeats = 10
        for r in range(repeats):
            score_accuracy = self.evaluate_model()

            score_accuracy = score_accuracy * 100.0

            print('>#%d: %.3f' % (r + 1, score_accuracy))
            scores.append(score_accuracy)
        # summarize results
        print(scores)
        m, s = mean(scores), std(scores)
        print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

        #Serialize model to JSON

        model_json = self.model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        #serialize weights to HDFS

        #model.save_weights("model.h5")
        self.model.save("lstm.h5")
        print("Saved model to disk")

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model

        #model = loaded_model.load_weights("model.h5")
        loadmodel = load_model("lstm.h5")
        print("Loaded model from disk")

        y_pred = loadmodel.predict(self.xtest_dataset)

        # 12 total classes

        labs = set()
        labs.add(0)
        labs.add(1)
        labs.add(2)
        labs.add(3)
        labs.add(4)
        labs.add(5)
        labs.add(6)
        labs.add(7)
        labs.add(8)
        labs.add(9)
        labs.add(10)
        labs.add(11)
        labs.add(12)
        preds = []
        new_test = []

        # converting one hot prediction  and real label to single interger value

        for i, p in enumerate(y_pred):
            preds.append(np.argmax(p))
            new_test.append(np.argmax(self.y_test[i]))

        y_pred = preds
        self.y_test = new_test

        #classification reports

        classes = ["Null class", "Standing still", "Sitting and relaxing", "Kying down", "Walking", "Climbing stairs", "Waist bends forward", "Frontal elevation of arms", "knees bending (crouching)", "Cycling", "Running", "Jump front & back"]

        print(classification_report(self.y_test, y_pred, target_names=classes))

        ## later

        ## load json and create model




        ##evaluate loaded model on test data



        #loadmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #loadmodel.fit(self.xtrain_valid_dataset, self.y_train, epochs=self.epoches, batch_size=self.batch_size, verbose=0)
        #loss, acc = loadmodel.evaluate(self.xtest_dataset, self.y_test, verbose=2)
        #print("%s: %.2f%%" % (loaded_model.metrics_names[1], acc[1]*100))

    def train(self):
        """Compiles and trains the model"""

    def evaluate(self):
        """Predicts resuts for the test dataset"""
