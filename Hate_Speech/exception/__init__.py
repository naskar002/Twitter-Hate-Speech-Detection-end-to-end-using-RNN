import os
import sys

def error_messeage_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script [{0}] at line [{1}]: {2}".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail):
        super().__init__(str(error))  # Use the original exception message
        self.error_message = error_messeage_detail(error, error_detail)  # Corrected

    def __str__(self):
        return self.error_message  # Return detailed error message instead of just the original message

    
'''import os
import sys

def error_messeage_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message
        self.error_detail = error_messeage_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message'''