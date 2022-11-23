import cv2
import mediapipe as mp

from lib.PoseConvertTools import PoseConvertTools


class BodyPoseDetector(PoseConvertTools):
	def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
		super().__init__()
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles
		self.mp_pose = mp.solutions.pose
		self.poseProcessor = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence, model_complexity=2)


	def ExtractPoints(self, results, h, w):
		points = []
		lm = results.pose_landmarks
		if lm is None:
			return None
		lmPose  = self.mp_pose.PoseLandmark
		points = self.convert2Node(lm, lmPose, h, w)
		return points


	def Detect(self, image):
		results = self.poseProcessor.process(image)
		h, w, c = image.shape
		points = self.ExtractPoints(results, h, w)
		return points


	def DrawPoints(self, image, results):
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		self.mp_drawing.draw_landmarks(
			image, results.pose_landmarks,
			self.mp_pose.POSE_CONNECTIONS,
			landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
		)
		return image

