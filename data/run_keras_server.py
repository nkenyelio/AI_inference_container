# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submit a request via Python:
#	python simple_request.py

import io
import os
import zipfile

import pandas as pd
import numpy as np
import flask
import numpy as np
from PIL import Image
from flask import json
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, model_from_json, load_model
# from keras.applications import imagenet_utils
# import the necessary packages
# from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
from werkzeug.utils import secure_filename, redirect

from dataloader import DataLoader_inference

from abc import ABC, abstractmethod
from utils.config import Config

# initialize our Flask application and the Keras model

UPLOAD_FOLDER = "data"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'log'}
app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = None

df = []


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    global df
    # model = ResNet50(weights="imagenet")
    # model = load_model("lstm.h5")
    model = tf.keras.models.load_model('lstm.h5')


def prepare_datapoint(subject_count, seq_size):
    # suppose that the subject mHealth data is one

    collect_save_data(subject_count, seq_size)
    read_dataX, read_dataY = load_data("./data/dataX.npy",
                                       "./data/dataY.npy")

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
    #file_name = 'mHealth_subject' + str(subject) + '.log'
    file_name = 'mHealth_subject1' + '.log'
    #file_pathr = os.path.join('data', file_name)
    file_pathr = "./data/" + "mHealth_subject" + ".log"

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


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("file"):
            # read the file provided
            file = flask.request.files["file"]
            if not file:
                return "No file"

            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flask.flash('No selected file')
                return redirect(flask.request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Write into file
            #subject = 1
            #filename = secure_filename(file.filename)
            #os.rename(filename, 'mHealth_subject' + str(subject) + '.log')
            #filename = 'mHealth_subject' + str(subject) + '.log'
            # file_name = 'mHealth_subject' + str(subject) + '.log'
            #os.path.join('data', filename)

            # DataLoader.read_subject(1)

            # preprocess the image and prepare it for classification
            X_test, Y_label = prepare_datapoint(1, 100)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            y_pred = model.predict(X_test)

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
                new_test.append(np.argmax(Y_label[i]))

            y_pred = preds
            Y_label = new_test

            # classification reports

            classes = ["Null class", "Standing still", "Sitting and relaxing", "Lying down", "Walking",
                       "Climbing stairs", "Waist bends forward", "Frontal elevation of arms",
                       "knees bending (crouching)", "Cycling", "Running", "Jump front & back"]

            report_class = classification_report(Y_label, y_pred, target_names=classes, output_dict=True)

            #results = json.dumps(report_class, indent=4)


            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions

            data["predictions"].append(report_class)
            # indicate that the request was a success

            data["success"] = True

    # return the data dictionary as a JSON response
    #return flask.jsonify(data)
    return flask.json.dumps(data, indent=4)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
