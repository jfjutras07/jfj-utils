import io
import sys
from utils import log_message, timeit

#--- Function : test_log_message ---
def test_log_message():
    # Test default INFO level
    captured_output = io.StringIO()
    sys.stdout = captured_output
    log_message("Test message")
    output = captured_output.getvalue().strip()
    sys.stdout = sys.__stdout__

    assert isinstance(output, str)
    assert "[INFO]" in output
    assert "Test message" in output

    # Test custom WARNING level
    captured_output = io.StringIO()
    sys.stdout = captured_output
    log_message("Warning message", level="WARNING")
    output = captured_output.getvalue().strip()
    sys.stdout = sys.__stdout__

    assert "[WARNING]" in output
    assert "Warning message" in output
