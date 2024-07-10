import numpy as np


def sigmoid (x):
	return 1 / (1 + np.exp(-x))


def costFunction (y, h):
	return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


class LogisticRegression:
	def __init__ (self, alpha=0.01, iterations=1000):
		self.alpha = alpha
		self.iterations = iterations
		self.theta = None
	
	def predict (self, X, theta):
		p = sigmoid(np.dot(X, theta))
		p[p >= 0.5] = 1
		p[p < 0.5] = 0
		return p
	
	def fit (self, X, y, theta, alpha=0.1, iterations=10000, cost=False):
		costH = []
		for i in range(iterations):
			z = np.dot(X, theta)
			h = sigmoid(z)
			g = np.dot(X.T, (h - y)) / len(y)
			theta -= alpha * g
			costH.append(costFunction(y, h))
			if cost:
				print(i, costH[i])
		return theta
