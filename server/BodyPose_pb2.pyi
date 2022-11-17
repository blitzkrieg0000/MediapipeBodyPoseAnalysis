from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ExtractBodyPoseRequest(_message.Message):
    __slots__ = ["frame"]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    frame: bytes
    def __init__(self, frame: _Optional[bytes] = ...) -> None: ...

class ExtractBodyPoseResponse(_message.Message):
    __slots__ = ["point"]
    POINT_FIELD_NUMBER: _ClassVar[int]
    point: bytes
    def __init__(self, point: _Optional[bytes] = ...) -> None: ...
