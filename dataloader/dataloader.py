# -*- coding: utf-8 -*-
"""Data Loader"""
import os
import zipfile

import jsonschema
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from configs.data_schema import SCHEMA

from utils.logger import get_logger

from sklearn.model_selection import train_test_split

LOG = get_logger('mHealth')


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(data_setX, data_setY):
        """Loads dataset from path the path for X and Y  """
        # return tfds.load(name="oxford_iiit_pet:3.*.*", data_dir="/home/ubuntu/PycharmProjects/aisummer/images")
        X_read = np.load(data_setX)
        Y_read = np.load(data_setY)
        return X_read, Y_read


    @staticmethod
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

    @staticmethod
    def preprocess_onhot_encoder(Y_train_set):
        """ Loads and process the one_hot encode
        :rtype: object
        """


        y_train = DataLoader.one_hot(Y_train_set, n_class=12)

        return y_train

    @staticmethod
    def preprocess_standardize_train_dataset(x_train_set, x_test_set):
        """ Loads and process the one_hot encode
        :rtype: object
        """

        ### standardize the X train and Y train  dataset
        X_train_dataset = DataLoader.standardize(x_train_set)
        X_test_dataset = DataLoader.standardize(x_test_set)

        return X_train_dataset, X_test_dataset

    @staticmethod
    def preprocess_train_validation_dataset(x_tr_set, y_tr_set):
        """ Loads and process the one_hot encode
        :rtype: object
        """

        ## train/validation
        X_train, X_valid, Y_train, Y_valid = train_test_split(x_tr_set, y_tr_set, test_size=0.4,
                                                              stratify=y_tr_set, random_state=456)

        return X_train, X_valid, Y_train, Y_valid


    @staticmethod
    def download_and_extract(mhealth_config):
        """## Download and extract data files"""
        url = mhealth_config.url
        LOG.info("Downloading..")
        r = requests.get(url)

        # Write into file
        open(mhealth_config.filename, 'wb').write(r.content)

        # Extract
        LOG.info('Extracting...')
        zip_h = zipfile.ZipFile(mhealth_config.filename, 'r')
        zip_h.extractall()
        zip_h.close()

        # Rename and remove zip
        os.rename(mhealth_config.removefolder, mhealth_config.rename)
        os.remove(mhealth_config.filename)

    @staticmethod
    def read_subject(subject):
        """## Read data per subject. Read measurements from a given subject"""
        file_name = 'mHealth_subject' + str(subject) + '.log'
        file_path = os.path.join('data', file_name)

        # Read file
        try:
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        except IOError:
            LOG.info("Data file does not exist!")

        # Remove data with null class (=0)
        df = df[df[23] != 0]

        return df

    @staticmethod
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

    @staticmethod
    def collect_save_data(subject_count, mblock_size):
        """ Collects all the data from all the subjects and writes in file """
        # Initiate lists
        X_ = []
        Y_ = []
        for s in range(1, subject_count + 1):
            # Read the data
            df = DataLoader.read_subject(s)

            # Split into blocks
            x, y = DataLoader.split_by_blocks(df, mblock_size)

            # Add to list
            X_.append(x)
            Y_.append(y)

        # Concatenate and save
        X = np.concatenate(X_, axis=0)
        Y = np.concatenate(Y_, axis=0)

        # Save
        np.save(os.path.join('data', 'dataX.npy'), X)
        np.save(os.path.join('data', 'dataY.npy'), Y)

    ## One-hot encoding
    @staticmethod
    def one_hot(labels, n_class=12):
        """ One-hot encoding """
        expansion = np.eye(n_class)
        y = expansion[:, labels - 1].T

        return y

    ## Standardize
    @staticmethod
    def standardize(X):
        """ Standardize by mean and std for each measurement channel"""
        return (X - np.mean(X, axis=0)[None, :, :]) / np.std(X, axis=0)[None, :, :]

    ## Get batches
    @staticmethod
    def get_batches(X, y, batch_size=100):
        """ Yield batches ffrom data """
        n_batches = len(X) // batch_size
        X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

        # Loop over batches and yield
        for b in range(0, len(X), batch_size):
            yield X[b:b + batch_size], y[b:b + batch_size]
