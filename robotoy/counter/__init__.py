from loguru import logger
import time
from robotoy.time_utils import PassiveTimer

class FpsCounter:
    def __init__(self):
        self.di = dict()

    def count(self, channel="default"):
        if channel not in self.di:
            self.di[channel] = [0, PassiveTimer()]
        self.di[channel][0] += 1

    def check(self, channel="default"):
        if channel not in self.di:
            self.di[channel] = [0, PassiveTimer()]
            return

        def fn():
            logger.info(f"{channel} fps: {self.di[channel][0]}")
            self.di[channel][0] = 0
        self.di[channel][1].try_act(
            time.time(),
            1.0, 
            fn
        )

    def count_and_check(self, channel="default"):
        self.count(channel)
        self.check(channel)
