import time
import grpc

import BodyPose_pb2 as rc
import BodyPose_pb2_grpc as rc_grpc
from lib.helpers import Converters

MAX_MESSAGE_LENGTH = 10*1024*1024

class BodyPoseClient():
    def __init__(self):
        self.channel = grpc.insecure_channel(
            'localhost:8000',
            options=[
                ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ],
            # compression=grpc.Compression.Gzip
        )
        self.stub = rc_grpc.BodyPoseStub(self.channel)
    

    def ExtractBodyPose(self, frame):
        frame = Converters.Frame2Bytes(frame)

        response = self.stub.ExtractBodyPose(rc.ExtractBodyPoseRequest(frame=frame))
        
        return response


    def Disconnect(self):
        self.channel.close()






