from enum import Enum
from typing import Any


class ResponseCodes(Enum):
    SUCCESS = 0
    WARNING = 1
    ERROR = 2
    INFO = 3
    NULL = 4
    NOT_FOUND = 5
    REQUIRED = 6
    UNSUFFICIENT = 7
    CONNECTION_ERROR = 8


class Response(object):
    def __init__(self, code:ResponseCodes, message:str=None, data:Any=None) -> None:
        self.code : ResponseCodes = code
        self.message = message
        self.data = data