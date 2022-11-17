import collections

import cv2
from lib.BodyPoseDetector import BodyPoseDetector
from lib.ExtraTools import ExtraTools

class CalculatePlayer(ExtraTools):
	def __init__(self):
		super().__init__()
		self.pose_detector = BodyPoseDetector()


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


	def pose_extractor(self, image):
		points = []
		points = self.pose_detector.GetBodyPose(image)

		if points is None:
			return image, [], [] #image, points, angles
		
		angles, image = self.getSpecialAngles(points, image)

		return image, points, angles


if "__main__" == __name__:

	cp = CalculatePlayer()

	input_video_path = "asset/video/train/roc/forehand/roc_1.7.mp4"
	cap = cv2.VideoCapture(input_video_path)

	
	while(True):
		ret, img = cap.read()
		if not ret:
			break

		canvas = img.copy()
		canvas, points, angles = cp.pose_extractor(img)

		cv2.imshow('', canvas)
		if cv2.waitKey(0) & 0xFF == ord("q"):
			break
	

	cap.release()
	cv2.destroyAllWindows()
