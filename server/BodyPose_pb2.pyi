from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExtractBodyPoseRequest(_message.Message):
    __slots__ = ["frame"]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    frame: bytes
    def __init__(self, frame: _Optional[bytes] = ...) -> None: ...

class ExtractBodyPoseResponse(_message.Message):
    __slots__ = ["Response"]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    Response: Response
    def __init__(self, Response: _Optional[_Union[Response, _Mapping]] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["Code", "Data", "Message"]
    class ResponseCodes(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    CODE_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_ERROR: Response.ResponseCodes
    Code: Response.ResponseCodes
    DATA_FIELD_NUMBER: _ClassVar[int]
    Data: bytes
    ERROR: Response.ResponseCodes
    INFO: Response.ResponseCodes
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    Message: str
    NOT_FOUND: Response.ResponseCodes
    NULL: Response.ResponseCodes
    REQUIRED: Response.ResponseCodes
    SUCCESS: Response.ResponseCodes
    UNSUFFICIENT: Response.ResponseCodes
    WARNING: Response.ResponseCodes
    def __init__(self, Code: _Optional[_Union[Response.ResponseCodes, str]] = ..., Message: _Optional[str] = ..., Data: _Optional[bytes] = ...) -> None: ...
