# A small work-around to execute the tests properly while using the src/tests layout in the
# repository. Not the most professional. As designed, pytest has to be executed from the
# source directory, not the test directory.
import sys

sys.path.append("./src")


# Imports
from timing import time


def test_time():
    def test_function(parameter: float) -> float:
        return parameter + 1

    test_parameter = 0
    assert type(time(function=test_function, parameter=test_parameter)) == float
