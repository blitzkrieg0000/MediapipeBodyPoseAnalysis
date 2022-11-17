import logging
import pickle
from concurrent import futures

import cv2
import grpc
import numpy as np

import BodyPose_pb2 as rc
import BodyPose_pb2_grpc as rc_grpc
from lib.CalculatePlayer import CalculatePlayer
from lib.Response import Response, ResponseCodes

logging.basicConfig(format='%(levelname)s - %(asctime)s => %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.NOTSET)


class BodyPoseServer(rc_grpc.BodyPoseServicer):
    def __init__(self):
        self.CalculatePlayer = CalculatePlayer()
    

    def CreateResponse(self, response:Response):
        return rc.Response(Code=rc.Response.ResponseCodes.Value(response.code.name), Message=response.message, Data=response.data)


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


    def ExtractBodyPose(self, request, context):
        frame = self.Bytes2Frame(request.frame)
        points, angles, canvas = self.CalculatePlayer.Process(frame)
        
        data = self.Obj2Bytes([points, angles, self.Frame2Bytes(canvas)]) 
        return rc.ExtractBodyPoseResponse(
            Response=self.CreateResponse(
                Response(ResponseCodes.SUCCESS, message="Producer Streaming Yapıyor...", data=data)
            )
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rc_grpc.add_BodyPoseServicer_to_server(BodyPoseServer(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()






