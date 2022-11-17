import base64
import hashlib
import json
import pickle
import re
import time
from json import JSONEncoder

import cv2
import numpy as np


#* Gelen Argümanlardan herhangi birisi None ise None döndür.
def checkNull(func):
    def wrapper(*args, **kwargs):
        if not all( [False for val in kwargs.values() if val is None]): return None
        if not all( [False for arg in args if arg is None]): return None
        return func(*args, **kwargs)
    return wrapper

#* Class Wrapper, class altındaki tüm methodlar için ilgili decoratorı tanımlar.
def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


@for_all_methods(checkNull)
class Converters():
    def __init__(self) -> None:
        pass
    

    @staticmethod
    def Bytes2Obj(bytes):
        if bytes != b'':
            return pickle.loads(bytes)
        return None


    @staticmethod
    def Obj2Bytes(obj):
        return pickle.dumps(obj)
    

    @staticmethod
    def Bytes2Frame(bytes_frame):
        nparr = np.frombuffer(bytes_frame, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    

    @staticmethod
    def Frame2Bytes(frame):
        res, encodedImg = cv2.imencode('.jpg', frame)
        return encodedImg.tobytes()
    

    @staticmethod
    def Frame2Base64(frame):
        etval, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode()


@for_all_methods(checkNull)
class Tools():
    EXCEPT_PREFIX = ['']
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def GetUID():
        return int.from_bytes(hashlib.md5(str(time.time()).encode("utf-8")).digest(), "little")

    @staticmethod
    def GenerateTopicName(prefix:str, id):
        prefix = prefix.strip()
        prefix = re.sub(r'\W+', '', prefix)
        prefix = prefix.encode('ascii', 'ignore').decode("utf-8")
        if prefix in Tools.EXCEPT_PREFIX:
            return prefix
        return f"{prefix}__{id}__{Tools.getUID()}"

    @staticmethod
    def DrawLines(cimage, points):
        for i, line in enumerate(points):
            if len(line)>0:
                cimage = cv2.line(cimage, ( int(line[0]), int(line[1]) ), ( int(line[2]), int(line[3]) ), (66, 245, 102), 3)
            if i==10:
                break
        return cimage

    @staticmethod
    def DrawCircles(cimage, fall_points, limit=1):
        for i, point in enumerate(fall_points):
            if len(point)>0:
                cimage = cv2.circle(cimage, (int(point[0]),int(point[1])), 5, (0,0,255), 1)
                cimage = cv2.circle(cimage, (int(point[0]),int(point[1])), 3, (255,255,255), 1)
            if i>=limit-1:
                break
        return cimage


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

@for_all_methods(checkNull)
class EncodeManager():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def Serialize(arr):
        return json.dumps(arr, cls=NumpyArrayEncoder)

    @staticmethod
    def Deserialize(arr):
        decodedArrays = json.loads(arr)
        return decodedArrays #np.asarray(decodedArrays)
