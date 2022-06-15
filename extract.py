# -*- coding: utf-8 -*-
"""Data Loader"""
import os
import zipfile

import jsonschema
from datetime import datetime, timedelta
import json
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from configs.data_schema import SCHEMA

from utils.logger import get_logger

from sklearn.model_selection import train_test_split

LOG = get_logger('mHealth')


def healthy_connection(mhealth_config):
    url = mhealth_config.url
    r = requests.get(url)
    if r.status_code == 200:
        return True
    else:
        return False

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

def main():

    current_date = datetime.now()
    previous_date = current_date - timedelta(days=1)
    
    if healthy_connection():
        r = download_and_extract(previous_date.date(), current_date.date(), all=True, limit=10000)
    else:
        raise Exception("Could not connect to mhealth data set.")

    with open("/results.json", "w") as f:
        json.dump(r, f)

if __name__ == '__main__':
    main()