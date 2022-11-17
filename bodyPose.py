import collections
import json
import math
import random
from dataclasses import dataclass

import cv2
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

class PoseConvertTools():
	def __init__(self):
		self.node = {}

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

class DetectPose(PoseConvertTools):
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
		self.pose_detector = DetectPose()

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
		angles["r_armpit"] = self.getAngle3D(points["RIGHT"]["elbow"], points["RIGHT"]["shoulder"], points["RIGHT"]["hip"], 1) #Right Armpit
		angles["l_armpit"] = self.getAngle3D(points["LEFT"]["elbow"], points["LEFT"]["shoulder"], points["LEFT"]["hip"], 1) #Left Armpit
		angles["r_elbow"] = self.getAngle3D(points["RIGHT"]["wrist"], points["RIGHT"]["elbow"], points["RIGHT"]["shoulder"], 1) #Right Elbow
		angles["l_elbow"] = self.getAngle3D(points["LEFT"]["wrist"], points["LEFT"]["elbow"], points["LEFT"]["shoulder"], 1) #Left Elbow
		angles["r_knee"] = self.getAngle3D(points["RIGHT"]["hip"], points["RIGHT"]["knee"], points["RIGHT"]["ankle"], 1) #Right Ankle
		angles["l_knee"] = self.getAngle3D(points["LEFT"]["hip"], points["LEFT"]["knee"], points["LEFT"]["ankle"], 1)  #Left Ankle
		#angles["dist_ankle"] = self.getDistance(points["LEFT"]["ankle"], points["RIGHT"]["ankle"])
		angles["leg_angle"] = self.getAngle3D(points["LEFT"]["ankle"], self.getMidPoint(points["LEFT"]["hip"], points["RIGHT"]["hip"]), points["RIGHT"]["ankle"])

		return angles, self.currentImage

	def drawAngles(self, image, angle, point_1, point_2, point_3):

		width, height = image.shape[0], image.shape[1]

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
		#TODO TRACKNET E ÇEVRİLECEK
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
			return image, [], [] #image, points, angles
		
		angles, image = self.getSpecialAngles(points, image)

		return image, points, angles


if "__main__" == __name__:
	#! SOURCE
	input_video_path = "video/train/roc/forehand/roc_1.7.mp4"
	cap = cv2.VideoCapture(input_video_path)


	cp = CalculatePlayer()

	while(True):
		ret, img = cap.read()
		if not ret:
			break

		canvas = img.copy()
		canvas, points, angles = cp.pose_extractor(img)

		cv2.imshow('', canvas)
		if cv2.waitKey(0) & 0xFF == ord("q"):
			break

	cv2.destroyWindow()
	cap.release()
