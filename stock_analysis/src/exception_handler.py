import sys
import logging

class CustomException(Exception):

    def __init__(self, error_detail:sys):
        super().__init__()
        _, self.cause , self.traceback  = sys.exc_info()

    def __str__(self):
        error_message =  (f"An error occurred in file '{self.traceback.tb_frame.f_code.co_filename}',"
              f" line {self.traceback.tb_lineno}, function {self.traceback.tb_frame.f_code.co_name} \n")

        return error_message + f"caused by: {self.cause}"
