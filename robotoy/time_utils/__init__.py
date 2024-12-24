import time

def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


def sleep_timer(time_interval, fn):
    last = time.time()
    while True:
        cur = time.time()
        till = max(cur, last + time_interval)
        precise_sleep(till - cur)
        last = till
        fn()


class PassiveTimer:
    def __init__(self):
        self.di = dict()

    def try_act(self, time, interval, act, channel='default'):
        if channel not in self.di:
            self.di[channel] = time + interval
            return
        if time >= self.di[channel]:
            act()
            self.di[channel] = max(time + 0.5 * interval, self.di[channel] + interval)


class Once:
    def __init__(self):
        self.s = set()

    def try_act(self, act, channel='default'):
        if channel not in self.s:
            self.s.add(channel)
            act()
            return


if __name__ == "__main__":
    once = Once()
    once.try_act(lambda: print('1'))
    once.try_act(lambda: print('1'))
    once.try_act(lambda: print('2'), '2')
    once.try_act(lambda: print('2'), '2')
    once.try_act(lambda: print('3'), '3')
    once.try_act(lambda: print('3'), '3')
    once.try_act(lambda: print('3'), '3')

    from robotoy.counter import FpsCounter
    cnt = FpsCounter()
    sleep_timer(0.100, lambda: print('ok') or cnt.count_and_check())

