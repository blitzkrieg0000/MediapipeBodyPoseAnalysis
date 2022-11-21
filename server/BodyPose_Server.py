import logging
from concurrent import futures
import time
import grpc

import BodyPose_pb2 as rc
import BodyPose_pb2_grpc as rc_grpc
from lib.CalculatePlayer import CalculatePlayer
from lib.helpers import Converters
from lib.Response import Response, ResponseCodes

logging.basicConfig(format='%(levelname)s - %(asctime)s => %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.NOTSET)


class BodyPoseServer(rc_grpc.BodyPoseServicer):
    def __init__(self):
        self.CalculatePlayer = CalculatePlayer()
    

    def CreateResponse(self, response:Response):
        return rc.Response(Code=rc.Response.ResponseCodes.Value(response.code.name), Message=response.message, Data=response.data)


    def ExtractBodyPose(self, request, context):
        frame = Converters.Bytes2Frame(request.frame)

        points, angles, canvas = self.CalculatePlayer.Process(frame)

        points = []
        angles = []
        
        data = Converters.Obj2Bytes([points, angles, canvas]) 
    
        res =  rc.ExtractBodyPoseResponse(
            Response=self.CreateResponse(
                Response(ResponseCodes.SUCCESS, message="Producer Streaming Yapıyor...", data=data)
            )
        )




        return res


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rc_grpc.add_BodyPoseServicer_to_server(BodyPoseServer(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()






