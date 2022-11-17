from lib.Point import Point


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


