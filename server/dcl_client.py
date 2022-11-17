from __future__ import print_function

import grpc

import detectCourtLines_pb2 as rc
import detectCourtLines_pb2_grpc as rc_grpc

import cv2


class DCLClient():
    def __init__(self):
        self.channel = grpc.insecure_channel('detectcourtlineservice:50021')
        self.stub = rc_grpc.detectCourtLineStub(self.channel)
    
    def img2bytes(self, image):
        res, encodedImg = cv2.imencode('.jpg', image)
        frame = encodedImg.tobytes()
        return frame

    def extractCourtLines(self, image):
        response = self.stub.extractCourtLines(rc.extractCourtLinesRequest(frame=image))
        return response.point

    def disconnect(self):
        self.channel.close()