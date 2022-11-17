from dataclasses import dataclass

import collections
import json
import math
import os
import random
import threading
import time
from queue import Queue

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
np.random.seed(42)


def softmax(x):
	return np.exp(x)/sum(np.exp(x))

@dataclass
class Point():
    x: float
    y: float
    z: float = 0.0


class FileVideoStream():
	def __init__(self, path, queueSize=4096):
		self.stream = cv2.VideoCapture(path)
		#Stream Settings
		"""
		0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
		1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
		2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
		3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
		4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
		5. CV_CAP_PROP_FPS Frame rate.
		6. CV_CAP_PROP_FOURCC 4-character code of codec.
		7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
		8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
		9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
		10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
		11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
		12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
		13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
		14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
		15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
		16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
		17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
		18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
		"""
		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
		self.stream.set(cv2.CAP_PROP_EXPOSURE, 0.1)
		self.stream.set(cv2.CAP_PROP_FPS, 30)
		self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H265'))

		self.stopped = False
		self.Q = Queue(maxsize=queueSize)
		self.t = threading.Thread(name="async_frame_reader", target=self.update, args=(), daemon=True)
	
	def start(self):
		self.t.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return

			if not self.Q.full():
				(grabbed, frame) = self.stream.read()
				if not grabbed:
					self.stop()
					return
				self.Q.put(frame, False)
				#print(self.Q.qsize())
				time.sleep(0.001)
	
	def read(self):
		return self.Q.get()

	def more(self):
		"""
		if self.Q.qsize() > 0:
		return True
		else:
		while True:
			if self.Q.qsize() > 0:
			break
		return True
		"""
		return self.Q.qsize() > 0

	def stop(self):
		self.stopped = True


class ExtraTools():
	def __init__(self):
		self.node = {}

	def preProcess(self, image):
		image = image[:,::-1]
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

	def convert2Node(self, lm, lmPose):
		self.node = {
			"CENTER":{
				"nose" : Point(lm.landmark[lmPose.NOSE].x, lm.landmark[lmPose.NOSE].y)
			},
			"LEFT":{
				"shoulder" : Point(lm.landmark[lmPose.LEFT_SHOULDER].x, lm.landmark[lmPose.LEFT_SHOULDER].y, lm.landmark[lmPose.LEFT_SHOULDER].z),
				"ankle" : Point(lm.landmark[lmPose.LEFT_ANKLE].x, lm.landmark[lmPose.LEFT_ANKLE].y, lm.landmark[lmPose.LEFT_ANKLE].z),
				"ear" : Point(lm.landmark[lmPose.LEFT_EAR].x, lm.landmark[lmPose.LEFT_EAR].y, lm.landmark[lmPose.LEFT_EAR].z),
				"elbow" : Point(lm.landmark[lmPose.LEFT_ELBOW].x, lm.landmark[lmPose.LEFT_ELBOW].y, lm.landmark[lmPose.LEFT_ELBOW].z),
				"eye" : Point(lm.landmark[lmPose.LEFT_EYE].x, lm.landmark[lmPose.LEFT_EYE].y, lm.landmark[lmPose.LEFT_EYE].z),
				"eyeInner" : Point(lm.landmark[lmPose.LEFT_EYE_INNER].x, lm.landmark[lmPose.LEFT_EYE_INNER].y, lm.landmark[lmPose.LEFT_EYE_INNER].z),
				"eyeOuter" : Point(lm.landmark[lmPose.LEFT_EYE_OUTER].x, lm.landmark[lmPose.LEFT_EYE_OUTER].y, lm.landmark[lmPose.LEFT_EYE_OUTER].z),
				"footIndex" : Point(lm.landmark[lmPose.LEFT_FOOT_INDEX].x, lm.landmark[lmPose.LEFT_FOOT_INDEX].y, lm.landmark[lmPose.LEFT_FOOT_INDEX].z),
				"heel" : Point(lm.landmark[lmPose.LEFT_HEEL].x, lm.landmark[lmPose.LEFT_HEEL].y, lm.landmark[lmPose.LEFT_HEEL].z),
				"hip" : Point(lm.landmark[lmPose.LEFT_HIP].x, lm.landmark[lmPose.LEFT_HIP].y, lm.landmark[lmPose.LEFT_HIP].z),
				"index" : Point(lm.landmark[lmPose.LEFT_INDEX].x, lm.landmark[lmPose.LEFT_INDEX].y, lm.landmark[lmPose.LEFT_INDEX].z),
				"knee" : Point(lm.landmark[lmPose.LEFT_KNEE].x, lm.landmark[lmPose.LEFT_KNEE].y, lm.landmark[lmPose.LEFT_KNEE].z),
				"pinky" : Point(lm.landmark[lmPose.LEFT_PINKY].x, lm.landmark[lmPose.LEFT_PINKY].y, lm.landmark[lmPose.LEFT_PINKY].z),
				"shoulder" : Point(lm.landmark[lmPose.LEFT_SHOULDER].x, lm.landmark[lmPose.LEFT_SHOULDER].y, lm.landmark[lmPose.LEFT_SHOULDER].z),
				"thumb" : Point(lm.landmark[lmPose.LEFT_THUMB].x, lm.landmark[lmPose.LEFT_THUMB].y, lm.landmark[lmPose.LEFT_THUMB].z),
				"wrist" : Point(lm.landmark[lmPose.LEFT_WRIST].x, lm.landmark[lmPose.LEFT_WRIST].y, lm.landmark[lmPose.LEFT_WRIST].z),
				"mouth" : Point(lm.landmark[lmPose.MOUTH_LEFT].x, lm.landmark[lmPose.MOUTH_LEFT].y, lm.landmark[lmPose.MOUTH_LEFT].z)
			},
			"RIGHT":{
				"shoulder" : Point(lm.landmark[lmPose.RIGHT_SHOULDER].x,lm.landmark[lmPose.RIGHT_SHOULDER].y, lm.landmark[lmPose.RIGHT_SHOULDER].z),
				"ankle" : Point(lm.landmark[lmPose.RIGHT_ANKLE].x,lm.landmark[lmPose.RIGHT_ANKLE].y, lm.landmark[lmPose.RIGHT_ANKLE].z),
				"ear" : Point(lm.landmark[lmPose.RIGHT_EAR].x,lm.landmark[lmPose.RIGHT_EAR].y, lm.landmark[lmPose.RIGHT_EAR].z),
				"elbow" : Point(lm.landmark[lmPose.RIGHT_ELBOW].x,lm.landmark[lmPose.RIGHT_ELBOW].y, lm.landmark[lmPose.RIGHT_ELBOW].z),
				"eye" : Point(lm.landmark[lmPose.RIGHT_EYE].x,lm.landmark[lmPose.RIGHT_EYE].y, lm.landmark[lmPose.RIGHT_EYE].z),
				"eyeInner" : Point(lm.landmark[lmPose.RIGHT_EYE_INNER].x,lm.landmark[lmPose.RIGHT_EYE_INNER].y, lm.landmark[lmPose.RIGHT_EYE_INNER].z),
				"eyeOuter" : Point(lm.landmark[lmPose.RIGHT_EYE_OUTER].x,lm.landmark[lmPose.RIGHT_EYE_OUTER].y, lm.landmark[lmPose.RIGHT_EYE_OUTER].z),
				"footIndex" : Point(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x,lm.landmark[lmPose.RIGHT_FOOT_INDEX].y, lm.landmark[lmPose.RIGHT_FOOT_INDEX].z),
				"heer" : Point(lm.landmark[lmPose.RIGHT_HEEL].x,lm.landmark[lmPose.RIGHT_HEEL].y, lm.landmark[lmPose.RIGHT_HEEL].z),
				"hip" : Point(lm.landmark[lmPose.RIGHT_HIP].x,lm.landmark[lmPose.RIGHT_HIP].y, lm.landmark[lmPose.RIGHT_HIP].z),
				"index" : Point(lm.landmark[lmPose.RIGHT_INDEX].x,lm.landmark[lmPose.RIGHT_INDEX].y, lm.landmark[lmPose.RIGHT_INDEX].z),
				"knee" : Point(lm.landmark[lmPose.RIGHT_KNEE].x,lm.landmark[lmPose.RIGHT_KNEE].y, lm.landmark[lmPose.RIGHT_KNEE].z),
				"pinky" : Point(lm.landmark[lmPose.RIGHT_PINKY].x,lm.landmark[lmPose.RIGHT_PINKY].y, lm.landmark[lmPose.RIGHT_PINKY].z),
				"shoulder" : Point(lm.landmark[lmPose.RIGHT_SHOULDER].x,lm.landmark[lmPose.RIGHT_SHOULDER].y, lm.landmark[lmPose.RIGHT_SHOULDER].z),
				"thumb" : Point(lm.landmark[lmPose.RIGHT_THUMB].x,lm.landmark[lmPose.RIGHT_THUMB].y, lm.landmark[lmPose.RIGHT_THUMB].z),
				"wrist" : Point(lm.landmark[lmPose.RIGHT_WRIST].x,lm.landmark[lmPose.RIGHT_WRIST].y, lm.landmark[lmPose.RIGHT_WRIST].z),
				"mouth" : Point(lm.landmark[lmPose.MOUTH_RIGHT].x,lm.landmark[lmPose.MOUTH_RIGHT].y, lm.landmark[lmPose.MOUTH_RIGHT].z)
			}
		}
		return self.node


class detectPose(ExtraTools):
	def __init__(self):
		super().__init__()
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles
		self.mp_pose = mp.solutions.pose
		self.poseProcessor = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

	def drawPoints(self, image, results):
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		self.mp_drawing.draw_landmarks(
			image, results.pose_landmarks,
			self.mp_pose.POSE_CONNECTIONS,
			landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
		)
		return image

	def extractPoints(self, results):
		points = []
		lm = results.pose_landmarks
		if lm is None:
			return
		lmPose  = self.mp_pose.PoseLandmark
		points = self.convert2Node(lm, lmPose)
		return points

	def getBodyPose(self, image):
		points = []

		results = self.poseProcessor.process(image)
		points = self.extractPoints(results)
		if points is None:
			return []

		return points

class CalculatePlayer(ExtraTools):
	def __init__(self):
		super().__init__()
		self.pose_detector = detectPose()

	def getMidPoint(self, first, second):
		point = Point(-1, -1, -1)
		try:
			Point((first.x + second.x) /2, (first.y + second.y) /2, (first.z + second.z) /2)
		except Exception as e:
			print("Distance Calculation: ", e)
		return point

	def getDistance(self, first, second):
		distance = -1
		distance = np.sqrt((first.x - second.x)**2 + (first.x - second.y)**2)
		return distance

	def getAngle(self, a, b, c):
		ang = math.degrees(math.atan2(c.y-b.y, c.x-b.x) - math.atan2(a.y-b.y, a.x-b.x))
		return ang

	def getAngle3D(self, A, B, C, draw=0):
		
		width, height = self.currentImage.shape[0], self.currentImage.shape[1]

		P1x = int(A.x * height)
		P1y = int(A.y * width)
		P1z = int(A.z)

		P2x = int(A.x * height)
		P2y = int(B.y * width)
		P2z = int(B.z)

		P3x = int(C.x * height)
		P3y = int(C.y * width)
		P3z = int(C.z)


		a = np.array([P1x,P1y,P1z])
		b = np.array([P2x,P2y,P2z])
		c = np.array([P3x,P3y,P3z])
		ba = a - b
		bc = c - b
		
		angle = -1
		try:
			cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
			angle = np.arccos(cosine_angle)*180.0 / np.pi
		except Exception as e:
			return -1
		
		if draw == 1:
			self.currentImage = self.drawAngles(self.currentImage, angle, A, B, C)

		return angle

	def pickColors(self, ):
		random_rgb = (int(random.random()*255), int(random.random()*255), int(random.random()*255))
		return random_rgb

	def getSpecialAngles(self, points, image):
		angles = collections.defaultdict(list)
		self.currentImage = image
		print(points)
		angles["r_armpit"] = self.getAngle3D(points["RIGHT"]["elbow"], points["RIGHT"]["shoulder"], points["RIGHT"]["hip"], 1) #Right Armpit
		angles["l_armpit"] = self.getAngle3D(points["LEFT"]["elbow"], points["LEFT"]["shoulder"], points["LEFT"]["hip"], 1) #Left Armpit
		angles["r_elbow"] = self.getAngle3D(points["RIGHT"]["wrist"], points["RIGHT"]["elbow"], points["RIGHT"]["shoulder"], 1) #Right Elbow
		angles["l_elbow"] = self.getAngle3D(points["LEFT"]["wrist"], points["LEFT"]["elbow"], points["LEFT"]["shoulder"], 1) #Left Elbow
		angles["r_knee"] = self.getAngle3D(points["RIGHT"]["hip"], points["RIGHT"]["knee"], points["RIGHT"]["ankle"], 1) #Right Ankle
		angles["l_knee"] = self.getAngle3D(points["LEFT"]["hip"], points["LEFT"]["knee"], points["LEFT"]["ankle"], 1)  #Left Ankle
		angles["dist_ankle"] = self.getDistance(points["LEFT"]["ankle"], points["RIGHT"]["ankle"])
		angles["leg_angle"] = self.getAngle3D(points["LEFT"]["ankle"], self.getMidPoint(points["LEFT"]["hip"], points["RIGHT"]["hip"]), points["RIGHT"]["ankle"])

		return angles, self.currentImage

	def drawAngles(self, image, angle, point_1, point_2, point_3):

		width, height = image.shape[0], image.shape[1]
		print(width, height)
		P1x = int(point_1.x * height)
		P1y = int(point_1.y * width)
		P2x = int(point_2.x * height)
		P2y = int(point_2.y * width)
		P3x = int(point_3.x * height)
		P3y = int(point_3.y * width)

		image = cv2.line(image, (P1x, P1y), (P2x, P2y), (255, 255, 255), 2)
		image = cv2.line(image, (P2x, P2y), (P3x, P3y), (255, 255, 255), 2)
		angle = float("%0.2f" % (angle))
		image = cv2.putText(image, str(angle), (P2x, P2y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 179, 245), 1, cv2.LINE_AA)

		return image

	def detectBall(self, image):
		bpj = []
		ball_position = self.yolov5_model([image])
		bpj = ball_position.pandas().xyxy[0].to_json(orient="records")
		"""
		bpj:json
			[{
				"xmin":1779.9069824219,"ymin":925.7464599609,
				"xmax":1830.6970214844,"ymax":974.7542724609,
				"confidence":0.6505390406,"class":32,"name":"sports ball"
			}]
		"""
		ball_coordinate = ()
		if len(bpj)>2:
			bpj = json.loads(bpj)
			ball_coordinate = (bpj[0]["xmin"],bpj[0]["ymax"])
		ball_coordinate = ((bpj[0]["xmin"]+bpj[0]["xmax"])/2, (bpj[0]["ymax"]+bpj[0]["ymin"])/2)
		return ball_coordinate
  
	def pose_extractor(self, image):
		points = []
		points = self.pose_detector.getBodyPose(image)

		if points is None:
			return image, [], [], [] #image, points, angles, ball_coordinate

		return points


import glob
if "__main__" == __name__:

	def calculate(data, poseEstimator):
		cap = FileVideoStream(data["file_path"]).start() #Start Thread
		
		angles = []
		bc = []
		
		time.sleep(1)
		while cap.more():
			frame = cap.read()
			image = poseEstimator.preProcess(frame)
			canvas_image = image.copy()
			ball_coordinate = []

			points = poseEstimator.run(image)
			angle, canvas_image = poseEstimator.getSpecialAngles(points, canvas_image)

			if len(points)==0:
				continue
			
			if len(angle)>0:
				angles.append(angle)

			print(f"Sağ Koltuk Altı Açısı: {angle['r_armpit']}")
			print(f"Sol Koltuk Altı Açısı: {angle['l_armpit']}")
			print(f"Sağ Dirsek Açısı: {angle['r_elbow']}")
			print(f"Sol Dirsek Açısı: {angle['l_elbow']}\n")
			print(f"Sağ Diz Açısı: {angle['r_knee']}")
			print(f"Sol Diz Açısı: {angle['l_knee']}")
			print(f"Bacak Açısı: {angle['leg_angle']}")

			if len(ball_coordinate)>1:
				bc.append(ball_coordinate)
			
			cv2.imshow('MediaPipe Pose', cv2.cvtColor(canvas_image, cv2.COLOR_BGR2RGB))

			if cv2.waitKey(1) & 0xFF == ord("q"):
				cap.stop()
				cv2.destroyAllWindows()
				break
		#parseData(data, angles)

	def parseData(data, angles):
		#jupyter notebook matplotlib setting
		#%matplotlib inline

		if len(angles)>0:
			# fig = plt.figure()
			# fig.set_dpi(1000)
		
			temp = collections.defaultdict(list)
			keys = list(angles[0].keys())
			for x in range(len(angles[0])):
				for i, item in enumerate(angles):
					temp.append(item[keys[x]])

				# t = np.linspace(0, len(temp[x])-1, len(temp[x]))
				# #temp[x] = softmax(temp[x])
		
				# random_rgb = (random.random(), random.random(), random.random())

				# plt.grid(True)
				# plt.rcParams["figure.figsize"] = (16,9)
				# plt.plot(t, temp[x], c=random_rgb)
				# plt.legend(keys[x])
				# if not os.path.exists(f"results/{file_name}"):
				# 	os.mkdir(f"results/{file_name}")
				# plt.title(keys[x])
				# plt.show()
				# fig.savefig(f'results/{file_name}/Angles_{keys[x]}.png', bbox_inches='tight')
				# plt.clf()
				# #temp = collections.defaultdict(list)

			htemp = np.hstack(temp)
			print("len:", len(htemp), htemp)


	poseEstimator = CalculatePlayer()
	for filename in glob.iglob("video/**/*.mp4", recursive=True):
		data = {}
		paths = filename.split("/")
		data["file_path"] = filename
		data["file_name"] = paths[-1]
		data["datasetType"] = paths[1]
		data["dataplayerType"] = paths[2]
		data["hitType"] = paths[3]
		calculate(data, poseEstimator)


"""
	#jupyter notebook matplotlib setting
	%matplotlib inline

	if len(angles)>0:
		fig = plt.figure()
		fig.set_dpi(1000)
		
		
		temp = collections.defaultdict(list)
		keys = list(angles[0].keys())
  
		for x in range(len(angles[0])):
			for i, item in enumerate(angles):
				temp[x].append(item[keys[x]])
			t = np.linspace(0, len(temp[x])-1, len(temp[x]))
			#temp[x] = softmax(temp[x])
	  
			random_rgb = (random.random(), random.random(), random.random())

			plt.grid(True)
			plt.rcParams["figure.figsize"] = (16,9)
			plt.plot(t, temp[x], c=random_rgb)
			plt.legend(keys[x])
			if not os.path.exists(f"results/{file_name}"):
				os.mkdir(f"results/{file_name}")
			plt.title(keys[x])
			plt.show()
			fig.savefig(f'results/{file_name}/Angles_{keys[x]}.png', bbox_inches='tight')
			plt.clf()
			#temp = collections.defaultdict(list)
"""