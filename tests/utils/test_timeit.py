import io
import sys
from utils import timeit

#--- Function : test_timeit ---
def test_timeit():
    # Test return value
    @timeit
    def add(a, b):
        return a + b

    result = add(2, 3)
    assert result == 5

    # Test logging of start and finish
    captured_output = io.StringIO()
    sys.stdout = captured_output

    @timeit
    def multiply(a, b):
        return a * b

    multiply(2, 3)
    output = captured_output.getvalue()
    sys.stdout = sys.__stdout__

    assert "Starting 'multiply'" in output
    assert "Finished 'multiply' in" in output
    assert "seconds." in output

    # Test propagation of exceptions
    @timeit
    def fail_func():
        raise ValueError("Intentional failure")

    try:
        fail_func()
        assert False  # Should not reach here
    except ValueError as e:
        assert str(e) == "Intentional failure"
