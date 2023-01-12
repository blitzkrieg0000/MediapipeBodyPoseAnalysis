from lib.Point import Point


class PoseConvertTools():
	def __init__(self):
		self.node = {}
	
	
	def convert2Node(self, lm, lmPose, nh = 1, nw = 1):
		self.node = {
			"CENTER":{
				"nose" : Point(lm.landmark[lmPose.NOSE].x, lm.landmark[lmPose.NOSE].y)
			},
			"LEFT":{
				"shoulder" : Point(lm.landmark[lmPose.LEFT_SHOULDER].x, lm.landmark[lmPose.LEFT_SHOULDER].y, lm.landmark[lmPose.LEFT_SHOULDER].z, nh, nw),
				"ankle" : Point(lm.landmark[lmPose.LEFT_ANKLE].x, lm.landmark[lmPose.LEFT_ANKLE].y, lm.landmark[lmPose.LEFT_ANKLE].z, nh, nw),
				"ear" : Point(lm.landmark[lmPose.LEFT_EAR].x, lm.landmark[lmPose.LEFT_EAR].y, lm.landmark[lmPose.LEFT_EAR].z, nh, nw),
				"elbow" : Point(lm.landmark[lmPose.LEFT_ELBOW].x, lm.landmark[lmPose.LEFT_ELBOW].y, lm.landmark[lmPose.LEFT_ELBOW].z, nh, nw),
				"eye" : Point(lm.landmark[lmPose.LEFT_EYE].x, lm.landmark[lmPose.LEFT_EYE].y, lm.landmark[lmPose.LEFT_EYE].z, nh, nw),
				"eyeInner" : Point(lm.landmark[lmPose.LEFT_EYE_INNER].x, lm.landmark[lmPose.LEFT_EYE_INNER].y, lm.landmark[lmPose.LEFT_EYE_INNER].z, nh, nw),
				"eyeOuter" : Point(lm.landmark[lmPose.LEFT_EYE_OUTER].x, lm.landmark[lmPose.LEFT_EYE_OUTER].y, lm.landmark[lmPose.LEFT_EYE_OUTER].z, nh, nw),
				"footIndex" : Point(lm.landmark[lmPose.LEFT_FOOT_INDEX].x, lm.landmark[lmPose.LEFT_FOOT_INDEX].y, lm.landmark[lmPose.LEFT_FOOT_INDEX].z, nh, nw),
				"heel" : Point(lm.landmark[lmPose.LEFT_HEEL].x, lm.landmark[lmPose.LEFT_HEEL].y, lm.landmark[lmPose.LEFT_HEEL].z, nh, nw),
				"hip" : Point(lm.landmark[lmPose.LEFT_HIP].x, lm.landmark[lmPose.LEFT_HIP].y, lm.landmark[lmPose.LEFT_HIP].z, nh, nw),
				"index" : Point(lm.landmark[lmPose.LEFT_INDEX].x, lm.landmark[lmPose.LEFT_INDEX].y, lm.landmark[lmPose.LEFT_INDEX].z, nh, nw),
				"knee" : Point(lm.landmark[lmPose.LEFT_KNEE].x, lm.landmark[lmPose.LEFT_KNEE].y, lm.landmark[lmPose.LEFT_KNEE].z, nh, nw),
				"pinky" : Point(lm.landmark[lmPose.LEFT_PINKY].x, lm.landmark[lmPose.LEFT_PINKY].y, lm.landmark[lmPose.LEFT_PINKY].z, nh, nw),
				"shoulder" : Point(lm.landmark[lmPose.LEFT_SHOULDER].x, lm.landmark[lmPose.LEFT_SHOULDER].y, lm.landmark[lmPose.LEFT_SHOULDER].z, nh, nw),
				"thumb" : Point(lm.landmark[lmPose.LEFT_THUMB].x, lm.landmark[lmPose.LEFT_THUMB].y, lm.landmark[lmPose.LEFT_THUMB].z, nh, nw),
				"wrist" : Point(lm.landmark[lmPose.LEFT_WRIST].x, lm.landmark[lmPose.LEFT_WRIST].y, lm.landmark[lmPose.LEFT_WRIST].z, nh, nw),
				"mouth" : Point(lm.landmark[lmPose.MOUTH_LEFT].x, lm.landmark[lmPose.MOUTH_LEFT].y, lm.landmark[lmPose.MOUTH_LEFT].z, nh, nw)
			},
			"RIGHT":{
				"shoulder" : Point(lm.landmark[lmPose.RIGHT_SHOULDER].x,lm.landmark[lmPose.RIGHT_SHOULDER].y, lm.landmark[lmPose.RIGHT_SHOULDER].z, nh, nw),
				"ankle" : Point(lm.landmark[lmPose.RIGHT_ANKLE].x,lm.landmark[lmPose.RIGHT_ANKLE].y, lm.landmark[lmPose.RIGHT_ANKLE].z, nh, nw),
				"ear" : Point(lm.landmark[lmPose.RIGHT_EAR].x,lm.landmark[lmPose.RIGHT_EAR].y, lm.landmark[lmPose.RIGHT_EAR].z, nh, nw),
				"elbow" : Point(lm.landmark[lmPose.RIGHT_ELBOW].x,lm.landmark[lmPose.RIGHT_ELBOW].y, lm.landmark[lmPose.RIGHT_ELBOW].z, nh, nw),
				"eye" : Point(lm.landmark[lmPose.RIGHT_EYE].x,lm.landmark[lmPose.RIGHT_EYE].y, lm.landmark[lmPose.RIGHT_EYE].z, nh, nw),
				"eyeInner" : Point(lm.landmark[lmPose.RIGHT_EYE_INNER].x,lm.landmark[lmPose.RIGHT_EYE_INNER].y, lm.landmark[lmPose.RIGHT_EYE_INNER].z, nh, nw),
				"eyeOuter" : Point(lm.landmark[lmPose.RIGHT_EYE_OUTER].x,lm.landmark[lmPose.RIGHT_EYE_OUTER].y, lm.landmark[lmPose.RIGHT_EYE_OUTER].z, nh, nw),
				"footIndex" : Point(lm.landmark[lmPose.RIGHT_FOOT_INDEX].x,lm.landmark[lmPose.RIGHT_FOOT_INDEX].y, lm.landmark[lmPose.RIGHT_FOOT_INDEX].z, nh, nw),
				"heer" : Point(lm.landmark[lmPose.RIGHT_HEEL].x,lm.landmark[lmPose.RIGHT_HEEL].y, lm.landmark[lmPose.RIGHT_HEEL].z, nh, nw),
				"hip" : Point(lm.landmark[lmPose.RIGHT_HIP].x,lm.landmark[lmPose.RIGHT_HIP].y, lm.landmark[lmPose.RIGHT_HIP].z, nh, nw),
				"index" : Point(lm.landmark[lmPose.RIGHT_INDEX].x,lm.landmark[lmPose.RIGHT_INDEX].y, lm.landmark[lmPose.RIGHT_INDEX].z, nh, nw),
				"knee" : Point(lm.landmark[lmPose.RIGHT_KNEE].x,lm.landmark[lmPose.RIGHT_KNEE].y, lm.landmark[lmPose.RIGHT_KNEE].z, nh, nw),
				"pinky" : Point(lm.landmark[lmPose.RIGHT_PINKY].x,lm.landmark[lmPose.RIGHT_PINKY].y, lm.landmark[lmPose.RIGHT_PINKY].z, nh, nw),
				"shoulder" : Point(lm.landmark[lmPose.RIGHT_SHOULDER].x,lm.landmark[lmPose.RIGHT_SHOULDER].y, lm.landmark[lmPose.RIGHT_SHOULDER].z, nh, nw),
				"thumb" : Point(lm.landmark[lmPose.RIGHT_THUMB].x,lm.landmark[lmPose.RIGHT_THUMB].y, lm.landmark[lmPose.RIGHT_THUMB].z, nh, nw),
				"wrist" : Point(lm.landmark[lmPose.RIGHT_WRIST].x,lm.landmark[lmPose.RIGHT_WRIST].y, lm.landmark[lmPose.RIGHT_WRIST].z, nh, nw),
				"mouth" : Point(lm.landmark[lmPose.MOUTH_RIGHT].x,lm.landmark[lmPose.MOUTH_RIGHT].y, lm.landmark[lmPose.MOUTH_RIGHT].z, nh, nw)
			}
		}

		return self.node


