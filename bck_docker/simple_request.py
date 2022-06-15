# import the necessary packages
import os

import requests

## initialize the keras REST API endpoint URL along with the input
# the mhleath data set

KERAS_REST_API_URL = "http://localhost:5000/predict"
MHEALTH_PATH = "./data/mHealth_subject.log"

# load the input mhealth and  construct the payload for the request

file = open(MHEALTH_PATH, "rb").read()
# os.path.join('data', file)
payload = {"file": file}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (key, result) in enumerate(r["predictions"]):
        print(key, '--')
        # Again iterate over the nested dictionary
        for subject, score in result.items():
            print(subject, ' : ', score)
# otherwise, the request failed
else:
    print("Requested failed")
