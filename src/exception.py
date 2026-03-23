

import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = self.get_error_message(error_message, error_detail)
        super().__init__(self.error_message)

    def get_error_message(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error in [{file_name}] at line [{line_number}] : {str(error_message)}"