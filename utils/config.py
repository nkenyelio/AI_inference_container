# -*- coding: utf-8 -*-
"""Config class"""

import json


class Config:
    """Config class which contains data, mhealth, train and model hyperparameters"""

    def __init__(self, data, mhealth, train, model):
        self.data = data
        self.mhealth = mhealth
        self.train = train
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.mhealth, params.train, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)