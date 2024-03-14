import sys
import logging
import traceback


class CustomException(Exception):
    def __init__(self, message="An error occurred"):
        super().__init__(message)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        self.traceback = exc_traceback
        self.cause = exc_value

    def __str__(self):
        if self.traceback and self.cause:
            # traceback information
            tb_details = traceback.extract_tb(self.traceback)
            last_call_stack = tb_details[-1] if tb_details else None
            file_name = last_call_stack.filename if last_call_stack else "Unknown file"
            line_num = last_call_stack.lineno if last_call_stack else "Unknown line"
            func_name = last_call_stack.name if last_call_stack else "Unknown function"

            error_message = (f"An error occurred in file '{file_name}',"
                             f" line {line_num}, function '{func_name}'\n"
                             f"Error Message: {str(self.cause)}")
            return error_message
        else:
            return super().__str__()
