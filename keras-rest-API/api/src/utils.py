# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submit a request via Python:
#	python simple_request.py


import os

import pandas as pd

import flask
import numpy as np

import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split

from config import (WORKER_SLEEP, REDIS_HOST, REDIS_PORT, REDIS_DB,
                    WEIGHTS_H5, INPUT_RAW_DATAX, INPUT_RAW_DATAY, FILE_PATH, LOG_DIR)

from sklearn.metrics import classification_report
from werkzeug.utils import secure_filename, redirect

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'log'}


def prepare_datapoint(subject_count, seq_size):
    # suppose that the subject mHealth data is one

    collect_save_data(subject_count, seq_size)
    read_dataX, read_dataY = load_data(INPUT_RAW_DATAX,
                                       INPUT_RAW_DATAY)

    xtrain_dataset, xtest_dataset, \
    ytrain_dataset, ytest_dataset = preprocess_train_test(read_dataX, read_dataY)

    xtrain_dataset, xtest_dataset = preprocess_standardize_train_dataset(xtrain_dataset,
                                                                         xtest_dataset)

    xtrain_valid_dataset, xvalid_dataset, \
    ytrain_valid_dataset, yvalid_dataset = preprocess_train_validation_dataset(
        xtrain_dataset,
        ytrain_dataset)

    y_train = preprocess_onhot_encoder(ytrain_valid_dataset)
    y_valid = preprocess_onhot_encoder(yvalid_dataset)
    y_test = preprocess_onhot_encoder(ytest_dataset)

    # return the test_dataset and label dataset
    return xtest_dataset, y_test


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def collect_save_data(subject_count, mblock_size):
    """ Collects all the data from all the subjects and writes in file """
    # Initiate lists
    X_ = []
    Y_ = []

    for s in range(1, subject_count + 1):
        # Read the data
        df = read_subject(s)

        # Split into blocks
        x, y = split_by_blocks(df, mblock_size)

        # Add to list
        X_.append(x)
        Y_.append(y)

    # Concatenate and save
    X = np.concatenate(X_, axis=0)
    Y = np.concatenate(Y_, axis=0)

    # Save
    np.save(os.path.join('data', 'dataX.npy'), X)
    np.save(os.path.join('data', 'dataY.npy'), Y)


def read_subject(subject):
    """## Read data per subject. Read measurements from a given subject"""
    # file_name = 'mHealth_subject' + str(subject) + '.log'
    file_name = 'mHealth_subject1' + '.log'
    # file_pathr = os.path.join('data', file_name)
    file_pathr = FILE_PATH

    df = []

    # Read file
    try:
        df = pd.read_csv(file_pathr, delim_whitespace=True, header=None)
    except IOError:
        print("Data file does not exist!")

    # Remove data with null class (=0)
    df = df[df[23] != 0]

    return df


def split_by_blocks(df, mblock_size):
    """ Split data from each subject into blocks of shorter length """
    # Channels
    n_channels = df.shape[1] - 1

    # Group by labels
    grps = df.groupby(23)

    # Create a list for concatenating
    X_ = []
    Y_ = []

    # Loop over groups (labels), reshape to tensor and concatenate
    for ig in range(1, len(grps) + 1, 1):
        df_ = grps.get_group(ig)

        # Data and targets
        y = pd.unique(df_[23].values)
        x = df_.drop(23, axis=1).to_numpy()

        n_blocks = len(x) // mblock_size
        x = x[:n_blocks * mblock_size]
        y = y[:n_blocks * mblock_size]

        x_tensor = x.reshape(-1, mblock_size, n_channels)

        # Append
        X_.append(x_tensor)
        Y_.append(np.array([y] * len(x_tensor), dtype=int).squeeze())

    # Concatenate and return
    X = np.concatenate(X_, axis=0)
    Y = np.concatenate(Y_, axis=0)

    return X, Y


def load_data(data_setX, data_setY):
    """Loads dataset from path the path for X and Y  """
    # return tfds.load(name="oxford_iiit_pet:3.*.*", data_dir="/home/ubuntu/PycharmProjects/aisummer/images")
    X_read = np.load(data_setX)
    Y_read = np.load(data_setY)
    return X_read, Y_read


def preprocess_train_test(X_dataset, Y_dataset):
    """ Loads and preprocess  the training dataset for X and Y
    :rtype: object
    """
    ## Train/test

    X_train_dataset, X_test_dataset, Y_train_dataset, Y_test_dataset = train_test_split(X_dataset, Y_dataset,
                                                                                        test_size=0.3,
                                                                                        stratify=Y_dataset,
                                                                                        random_state=123)

    return X_train_dataset, X_test_dataset, Y_train_dataset, Y_test_dataset


def preprocess_onhot_encoder(Y_train_set):
    """ Loads and process the one_hot encode
    :rtype: object
    """

    y_train = one_hot(Y_train_set, n_class=12)

    return y_train


def preprocess_standardize_train_dataset(x_train_set, x_test_set):
    """ Loads and process the one_hot encode
    :rtype: object
    """

    ### standardize the X train and Y train  dataset
    X_train_dataset = standardize(x_train_set)
    X_test_dataset = standardize(x_test_set)

    return X_train_dataset, X_test_dataset


def preprocess_train_validation_dataset(x_tr_set, y_tr_set):
    """ Loads and process the one_hot encode
    :rtype: object
    """

    ## train/validation
    X_train, X_valid, Y_train, Y_valid = train_test_split(x_tr_set, y_tr_set, test_size=0.4,
                                                          stratify=y_tr_set, random_state=456)

    return X_train, X_valid, Y_train, Y_valid


## One-hot encoding

def one_hot(labels, n_class=12):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels - 1].T

    return y


## Standardize

def standardize(X):
    """ Standardize by mean and std for each measurement channel"""
    return (X - np.mean(X, axis=0)[None, :, :]) / np.std(X, axis=0)[None, :, :]

    ## Get batches


def get_batches(X, y, batch_size=100):
    """ Yield batches ffrom data """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b + batch_size], y[b:b + batch_size]


def addUploadedFiles():
    """ Added the uploaded files"""
