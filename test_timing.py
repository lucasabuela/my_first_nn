from timing import time


def test_time():
    def test_function(parameter: float) -> float:
        return parameter + 1

    test_parameter = 0
    assert type(time(function=test_function, parameter=test_parameter)) == float
