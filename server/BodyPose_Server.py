import logging
from concurrent import futures
import pickle

import grpc
import BodyPose_pb2 as rc
import BodyPose_pb2_grpc as rc_grpc

logging.basicConfig(format='%(levelname)s - %(asctime)s => %(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=logging.NOTSET)

class BodyPoseServer(rc_grpc.BodyPoseServicer):

    def __init__(self):
        pass
    
    def obj2bytes(self, obj):
        return pickle.dumps(obj)

    def ExtractBodyPose(self, request, context):


        return rc.ExtractBodyPoseResponse(point="")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rc_grpc.add_BodyPoseServicer_to_server(BodyPoseServer(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()