import cv2
import mediapipe as mp

from lib.PoseConvertTools import PoseConvertTools


class BodyPoseDetector(PoseConvertTools):
	def __init__(self):
		super().__init__()
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_drawing_styles = mp.solutions.drawing_styles
		self.mp_pose = mp.solutions.pose
		self.poseProcessor = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


	def DrawPoints(self, image, results):
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		self.mp_drawing.draw_landmarks(
			image, results.pose_landmarks,
			self.mp_pose.POSE_CONNECTIONS,
			landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
		)
		return image


	def ExtractPoints(self, results):
		points = []
		lm = results.pose_landmarks
		if lm is None:
			return
		lmPose  = self.mp_pose.PoseLandmark
		points = self.convert2Node(lm, lmPose)
		return points


	def GetBodyPose(self, image):
		results = self.poseProcessor.process(image)
		points = self.ExtractPoints(results)
		return points
