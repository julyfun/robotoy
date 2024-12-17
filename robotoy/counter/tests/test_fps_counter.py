import sys
import time
import os
import random
if 1:
    # add parent
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from counter import FpsCounter
    from time_utils import precise_sleep

client = ['zig', 'zag', 'rust']

cnt = FpsCounter()

while True:
    # 随机选一个 client
    for name in client:
        cnt.count_and_check(name)

    precise_sleep(0.01)
