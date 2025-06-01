"""
As the performance will probably be a bottleneck, it is important to know which functions and steps
take up the most time. This script provides a general timing function.
"""

from typing import Callable
from timeit import timeit


def time(function: Callable, nb_repetitions: int = 10, *args, **kwargs):
    """
    This function should measure the average time, on a provided number of repetitions, of the
    provided function with the provided arguments.

    Args:
        function (Callable): the function to be timed.
        nb_repetitions (int): the number of measures to be taken, to average the noise. Default to 10.
        *args : the args of the function.

    Returns:
        average_run_time (float): average run time in s.
    """

    def function_tested():
        return function(*args, **kwargs)

    summed_run_time = timeit(
        stmt=function_tested,
        number=nb_repetitions,
        globals=globals(),
    )
    average_run_time = summed_run_time / nb_repetitions
    return average_run_time
