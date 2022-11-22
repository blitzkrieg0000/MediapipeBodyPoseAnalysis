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

MAX_MESSAGE_LENGTH = 10*1024*1024

class BodyPoseServer(rc_grpc.BodyPoseServicer):
    def __init__(self):
        self.CalculatePlayer = CalculatePlayer()
    

    def CreateResponse(self, response:Response):
        return rc.Response(Code=rc.Response.ResponseCodes.Value(response.code.name), Message=response.message, Data=response.data)


    def ExtractBodyPose(self, request, context):
        frame = Converters.Bytes2Obj(request.frame)

        points, angles, canvas = self.CalculatePlayer.Process(frame)

        data = Converters.Obj2Bytes([points, angles, canvas]) 
        # with open("result/dump.txt", "w") as f:
        #     f.write(str(data))

        responseData =  rc.ExtractBodyPoseResponse(
            Response=self.CreateResponse(
                Response(ResponseCodes.SUCCESS, message="...", data=data)
            )
        )

        return responseData



def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
        ],
        # compression=grpc.Compression.Gzip
    )

    rc_grpc.add_BodyPoseServicer_to_server(BodyPoseServer(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()






