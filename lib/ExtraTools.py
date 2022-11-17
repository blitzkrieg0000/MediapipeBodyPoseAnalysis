import math
import random
import numpy as np
np.random.seed(42)
from lib.Point import Point

class ExtraTools():
	def __init__(self) -> None:
		pass

	def softmax(x):
		return np.exp(x)/sum(np.exp(x))


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
