"""
stopwatch is a very simple Python module for measuring time.
Great for finding out how long code takes to execute.

>>> import stopwatch
>>> t = stopwatch.Timer()
>>> t.elapsed
3.8274309635162354
>>> print t
15.9507198334 sec
>>> t.stop()
30.153270959854126
>>> print t
30.1532709599 sec

Decorator exists for printing out execution times:
>>> from stopwatch import clockit
>>> @clockit
    def mult(a, b):
        return a * b
>>> print mult(2, 6)
mult in 1.38282775879e-05 sec
6
"""

import time
import datetime
import unittest


class Timer(object):
    def __init__(self):
        self.__stopped = None
        self.__start = self.__time()

    def restart(self):
        """Starts again with clear timer begin and end."""
        self.__stopped = None
        self.__start = self.__time()

    def stop(self):
        """Stops the clock permanently for the instance of the Timer.
        Returns the time at which the instance was stopped.
        """
        self.__stopped = self.__last_time()
        return self.elapsed

    @property
    def elapsed(self):
        """The number of seconds since the current time that the Timer
        object was created.  If stop() was called, it is the number
        of seconds from the instance creation until stop() was called.
        """
        return self.__last_time() - self.__start

    @property
    def start_time(self):
        """The time at which the Timer instance was created."""
        return self.__start

    @property
    def stop_time(self):
        """The time at which stop() was called, or None if stop was
        never called.
        """
        return self.__stopped

    def __last_time(self):
        """Return the current time or the time at which stop() was call,
        if called at all.
        """
        if self.__stopped is not None:
            return self.__stopped
        return self.__time()

    def __time(self):
        """Wrapper for time.time() to allow unit testing."""
        return time.time()

    def __str__(self):
        """Nicely format the elapsed time"""
        return str(self.elapsed) + " sec"

    def __enter__(self):
        """possible use: with Timer() as t:"""
        self.restart()
        return self

    def __exit__(self, *args):
        self.stop()


def clockit(func):
    """Function decorator that times the evaluation of *func* and prints the
    execution time.
    """

    def new(*args, **kw):
        timer = Timer()
        retval = func(*args, **kw)
        timer.stop()
        print("%s in %s" % (func.__name__, timer))
        del timer
        return retval

    return new


class OverTimer(Timer):
    def __init__(self, threshold):
        """starts from creation with threshold in seconds."""
        super(OverTimer, self).__init__()
        self.threshold = float(threshold)

    def overtime(self):
        """Indicate if time is over threshold."""
        if self.threshold <= 0:
            return True
        return self.elapsed > self.threshold

    def remind_to_overtime(self):
        return max(0, self.threshold - self.elapsed)

    def set_threshold(self, threshold):
        """Set new threshold."""
        self.threshold = float(threshold)


class PointTimer(OverTimer):
    def __init__(self, point_datetime):
        dt_now = datetime.datetime.now()
        if point_datetime < dt_now:
            threshold = -(dt_now - point_datetime).seconds
        else:
            threshold = (point_datetime - dt_now).seconds
        super(PointTimer, self).__init__(threshold)

    def set_threshold(self, threshold):
        """Set new threshold is disallowed."""
        raise NotImplementedError("PointTimer can has not set_threshold")

    def restart(self):
        """Starts again is disallowed."""
        raise NotImplementedError("PointTimer can has not restart")


class PointTimerTest(unittest.TestCase):
    def test_initialy_passed(self):
        ptimer = PointTimer(
            datetime.datetime(year=2017, month=9, day=25, hour=8, minute=40)
        )
        self.assertTrue(ptimer.overtime())


if __name__ == "__main__":
    unittest.main()
