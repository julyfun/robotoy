from loguru import logger
import time

class FpsCounter:
    def __init__(self):
        self.time_di = dict()
        self.count_di = dict()

    def count(self, name):
        if name not in self.time_di:
            self.time_di[name] = time.time()
            self.count_di[name] = 0
        self.count_di[name] += 1

    def check(self, name):
        t = time.time()
        if name not in self.time_di:
            self.time_di[name] = t
            self.count_di[name] = 0
            return
        if t - self.time_di[name] > 1:
            logger.info(f"{name} fps: {self.count_di[name]}")
            self.time_di[name] = t
            self.count_di[name] = 0

    def count_and_check(self, name):
        self.count(name)
        self.check(name)
