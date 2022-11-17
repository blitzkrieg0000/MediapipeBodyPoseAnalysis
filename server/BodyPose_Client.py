from __future__ import print_function

import pickle

import cv2
import grpc
import numpy as np

import BodyPose_pb2 as rc
import BodyPose_pb2_grpc as rc_grpc
from lib.helpers import Converters

class BodyPoseClient():
    def __init__(self):
        self.channel = grpc.insecure_channel('localhost:8000')
        self.stub = rc_grpc.BodyPoseStub(self.channel)
    

    def ExtractBodyPose(self, frame):
        frame = Converters.Frame2Bytes(frame)
        response = self.stub.ExtractBodyPose(rc.ExtractBodyPoseRequest(frame=frame))
        return response


    def Disconnect(self):
        self.channel.close()


def Bytes2Frame(image):
    nparr = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


if "__main__" == __name__:
    client = BodyPoseClient()

    input_video_path = "server/asset/video/train/roc/forehand/roc_1.7.mp4"
    cap = cv2.VideoCapture(input_video_path)
    
    while(True):
        ret, frame = cap.read()
        if not ret:
            break

        responseData = client.ExtractBodyPose(frame)
        results = pickle.loads(responseData.Response.Data)
        
        cv2.imshow('', Bytes2Frame(results[2]) if results[2] is not None else frame )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
	
    cap.release()
    cv2.destroyAllWindows()





