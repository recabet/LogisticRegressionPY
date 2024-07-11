import numpy as np
def sigmoid (x):
	return 1 / (1 + np.exp(-x))


class my_LogisticRegression:
	def __init__ (self, alpha=0.01, iterations=10000):
		self.alpha = alpha
		self.iterations = iterations
		self.theta = None
	
	def cost_function (self, y, h):
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
	
	def fit (self, X, y, cost=False):
		self.theta = np.zeros(X.shape[1])
		for i in range(self.iterations):
			z = np.dot(X, self.theta)
			h = sigmoid(z)
			g = np.dot(X.T, (h - y)) / len(y)
			self.theta -= self.alpha * g
			if cost and i % 1000 == 0:
				print(f"Iteration: {i}, Cost: {self.cost_function(y, h)}")
		
		return self.theta
	
	def predict (self, X):
		p = sigmoid(np.dot(X, self.theta))
		p[p >= 0.5] = 1
		p[p < 0.5] = 0
		return p