import time
import cv2
from BodyPose_Client import BodyPoseClient
from lib.helpers import Converters
import numpy as np

if "__main__" == __name__:
    bodyPoseClient = BodyPoseClient()
    input_video_path = "server/asset/video/test.mp4"
    cap = cv2.VideoCapture(input_video_path)


    tic = 0
    canvas = None
    frame_counter = 0
    fps = 0
    fps_average = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.imread("server/asset/image/temp_cropped.jpg")

        response = bodyPoseClient.ExtractBodyPose(frame)
        

        if response.Data is not None:
            points, angles, canvas = Converters.Bytes2Obj(response.Data)
        

        if frame_counter>0:
            toc = time.time()
            fps += 1/(toc-tic)
            fps_average = int(fps / frame_counter)
            tic = time.time()
        frame_counter+=1

        print(canvas)
        if canvas is not None:
            canvas = cv2.putText(canvas, f"FPS: {fps_average}", (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255))
            frame = canvas


        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        

    cap.release()
    cv2.destroyAllWindows()