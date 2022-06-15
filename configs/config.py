# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "path": "data",
        "image_size": 128,
        "load_with_info": True
    },
    "mhealth": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
        "filename": "MHEALTHDATASET.zip",
        "removefolder": "MHEALTHDATASET",
        "rename": "data",
        "mblock_size": 100,
        "mchannel": 23,
        "subject_count": 10,
        "n_class": 12,
        "mbatch_size": 100,
        "path_readX": "./data/dataX.npy",
        "path_readY": "./data/dataY.npy"
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epoches": 15,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}