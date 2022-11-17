from __future__ import print_function

import pickle

import cv2
import grpc
import numpy as np

import BodyPose_pb2 as rc
import BodyPose_pb2_grpc as rc_grpc


class BodyPoseClient():
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:8000')
        self.stub = rc_grpc.BodyPoseStub(self.channel)
    

    def Bytes2Obj(self, bytes):
        return pickle.loads(bytes)


    def Obj2Bytes(self, obj):
        return pickle.dumps(obj)


    def Bytes2Frame(self, image):
        nparr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
        
        
    def Frame2Bytes(self, image):
        res, encodedImg = cv2.imencode('.jpg', image)
        frame = encodedImg.tobytes()
        return frame


    def ExtractBodyPose(self, frame):
        frame = self.Frame2Bytes(frame)
        response = self.stub.ExtractBodyPose(rc.ExtractBodyPoseRequest(frame=frame))
        return response


    def Disconnect(self):
        self.channel.close()


def Bytes2Frame(image):
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


if '__main__' == __name__:
    client = BodyPoseClient()
    img = cv2.imread("server/asset/test/1.png")
    responseData = client.ExtractBodyPose(img)

    results = pickle.loads(responseData.Response.Data)
    print(results[2])




