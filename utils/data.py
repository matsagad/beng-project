from pathlib import Path
from utils.process import get_tf_data
import os
import pickle
import re
import sys


class DataSingleton:
    _instance = None

    def __init__(self):
        (
            self.data,
            self.origin,
            self.time_delta,
            self.tf_names,
        ) = DataSingleton._get_tf_data()
        self.camel_to_snake = re.compile(r"(?<!^)(?=[A-Z])")

    def _get_tf_data(**kwargs):
        args = sys.argv[1:]
        if not args:
            return get_tf_data(**kwargs)
        return get_tf_data(cache_folder=args[0], **kwargs)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class ClassWithData:
    def __init__(self):
        self.__dict__.update(DataSingleton.instance().__dict__)
        self.interval = 30

        self.default_sim_args = {
            "exogenous_data": self.data,
            "tau": self.time_delta,
            "realised": True,
            "replicates": 10,
        }

        self.default_est_args = {
            "origin": self.origin,
            "interval": self.interval,
            "replicates": 10,
            "classifier_name": "naive_bayes",
        }

        self.default_pip_args = {
            **self.default_sim_args,
            **self.default_est_args,
            "classifier_name": "naive_bayes",
        }

        self.CACHE_FOLDER = "cache/latest"

        _name = self.camel_to_snake.sub("_", self.__class__.__name__).lower()
        self.SAVE_FOLDER = f"{self.CACHE_FOLDER}/{_name}/saves"
        self.CACHING_FOLDER = f"{self.CACHE_FOLDER}/{_name}/cache"

        for folder in (self.SAVE_FOLDER, self.CACHING_FOLDER):
            Path(folder).mkdir(exist_ok=True, parents=True)

    def unpickle(self, fname: str):
        object = None
        if os.path.isfile(fname):
            with open(fname, "rb") as f:
                object = pickle.load(f)
        else:
            print(f"File {fname} not found.")
        return object

    def pickle(self, fname: str, object: any):
        with open(fname, "wb") as f:
            pickle.dump(object, f)
        