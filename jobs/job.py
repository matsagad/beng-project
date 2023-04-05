from typing import Dict
from abc import ABC, abstractmethod
import time
import sys

class Job(ABC):
    def __init__(self, verbose=True):
        self.verbose = verbose

    def timed(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            print(f"Job ran for {time.time() - start:.3f}s")
        return wrapper
    
    @timed
    @abstractmethod
    def run(self, args: Dict[str, any]) -> None:
        pass

    @abstractmethod
    def on_interrupted(self, *args, **kwargs) -> None:
        sys.exit()
