from utils.process import get_tf_data
import pickle
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
        self.CACHE_FOLDER = "cache/latest"
    
    def unpickle(self, fname):
        object = None
        with open(fname, "rb") as f:
            object = pickle.load(f)
        return object