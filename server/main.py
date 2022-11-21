import time
import cv2
import pickle
from BodyPose_Client import BodyPoseClient
from lib.helpers import Converters



if "__main__" == __name__:
    client = BodyPoseClient()
    input_video_path = "server/asset/video/train/roc/forehand/roc_1.7.mp4"
    cap = cv2.VideoCapture(input_video_path)
    
    while(True):
        ret, frame = cap.read()
        if not ret:
            continue

        responseData = client.ExtractBodyPose(frame)
        
        data = []
        if responseData.Response.Data is not None:
            data = Converters.Bytes2Obj(responseData.Response.Data)
            frame = data[2]

        cv2.imshow('', frame )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()