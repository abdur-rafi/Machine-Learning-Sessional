import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

np.random.seed(5)



def loadData(path):
	data = pd.read_csv(path, header=None)
	return data


def pcaUsingSVD(data, components):
	# calculate svd
	U, S, V = np.linalg.svd(data, full_matrices=True)
	dataReduced = np.dot(data, V[:, :components])
	return dataReduced


def pcaUsingCoVarMatrix(data, components):
	covariance = np.cov(data, rowvar=False)
	eigen_values, eigen_vectors = np.linalg.eig(covariance)
	sortedIndex = np.argsort(eigen_values)
	# print(sortedIndex)
	# print(eigen_values)
	# print(eigen_vectors)
	principalVectors = eigen_vectors[:, sortedIndex[0:components]]
	pca = np.dot(data, principalVectors)
	return pca


def pcaUsingLib(data, components):
	from sklearn.decomposition import PCA
	pca = PCA(n_components=components)
	pca.fit(data)
	return pca.transform(data)

def scatterPlot(data, title, imgPath):
	plt.scatter(data[:, 0], data[:, 1])
	plt.grid(True)
	plt.title(title)
	if imgPath is not None:
		plt.savefig(imgPath)
	else:
		plt.show()


def pca(data):
	data = (data - data.mean()) / data.std()
	components = 2
	return pcaUsingCoVarMatrix(data, components)



class MultiVariateGaussian:

	def __init__(self, dimension, mean, covariance):
		self.dimension = dimension
		self.mean = mean
		if mean is None:
			self.mean = np.random.rand(dimension, 1)
		self.covariance = covariance
		if covariance is None:
			while True:
				self.covariance = np.random.rand(dimension, dimension)
				det = np.linalg.det(self.covariance)
				if det > 0:
					break
		

	def getProbabilityBatch(self, data):
		assert data.shape[1] == self.dimension

		data = np.expand_dims(data, axis=2) - self.mean
		dataT = np.transpose(data, (0, 2, 1))
		expo = dataT @ np.linalg.inv(self.covariance) @ data

		# print("data", data.shape)
		# print("dataT", dataT.shape)
		# print("expo", expo.shape)

		assert expo.shape == (data.shape[0], 1, 1)
		expo = expo * -0.5
		expo = np.squeeze(expo, axis=2)
		expo[expo > 100] = 100
		prob = np.exp(expo)
		prob = prob / np.sqrt(np.linalg.det(self.covariance))
		prob = prob / np.power(2 * np.pi, self.dimension / 2)
		return prob
	
		


	# x is a vector
	def getProbability(self, x):
		assert x.shape == (self.dimension, 1)

		xSubMean = x - self.mean
		exponent = xSubMean.T @ np.linalg.inv(self.covariance) @ xSubMean
		exponent = exponent * -0.5
		# print("expo", exponent)
		prob = np.exp(exponent)
		# print("prob",prob)
		det = np.linalg.det(self.covariance)
		# print("det", det)
		prob = prob / np.sqrt(det)
		prob = prob / np.power(2 * np.pi, self.dimension / 2)
		return prob[0,0]        
	
	def updateMeanAndCovariance(self, mean, covariance):
		self.mean = mean
		self.covariance = covariance
# (2, 1)
# (n,1,2) x (n,2,2) x (n,2,1)    
# (1,2) x (2,2) x (2,1)    
# (n,2) x (2,2) x (2,n)    

class EM:
	def __init__(self, dimension, components):
		self.dimension = dimension
		self.components = components
		self.gModels = []
		for i in range(components):
			self.gModels.append(MultiVariateGaussian(dimension, None, None))

		self.weights = np.random.rand(components, 1) 
		self.weights = self.weights / np.sum(self.weights)
		
	

	def EStepSingleModel(self, data, index):

		model = self.gModels[index]
		prob = model.getProbabilityBatch(data)
		# prob = prob / np.sum(prob)
		return prob
	

	def MStepSingleModel(self, data, probVector, index):
		model = self.gModels[index]
		n = np.sum(probVector)
		mean = np.multiply(probVector.T, data.T)
		mean = np.expand_dims(np.sum(mean, axis=1), 1)
		mean = mean / n

		data = np.expand_dims(data, axis=2) - mean
		dataT = np.transpose(data, (0, 2, 1))

		covariance = data @ dataT
		covariance = np.multiply(covariance, np.expand_dims(probVector, axis = 2))
		# while True:
		# 	if np.linalg.det(covariance) > 0:
		# 		break
		# 	covariance = np.random.rand(self.dimension, self.dimension)
		
		covariance = np.sum(covariance, axis=0) / n
		if np.linalg.det(covariance) <= 0:
			# print("covariance", covariance)
			while True:
				covariance = np.random.rand(self.dimension, self.dimension)
				det = np.linalg.det(covariance)
				if det > 0:
					break

		model.updateMeanAndCovariance(mean, covariance)
		

		index = self.gModels.index(model)
		self.weights[index] = n / data.shape[0]


	def OneStep(self, data):
		probVectors = []
		for i, model in enumerate(self.gModels):
			probVector = self.EStepSingleModel(data, i) * self.weights[i]
			probVectors.append(probVector)
		probVectors = np.concatenate(probVectors, axis=1)
		assert probVectors.shape == (data.shape[0], self.components)
		probVectors = probVectors / np.sum(probVectors, axis=1, keepdims=True)

		for i, model in enumerate(self.gModels):
			probVector = np.expand_dims(probVectors[:, i], axis=1)
			self.MStepSingleModel(data, probVector, i)
			# self.MStepSingleModel(data, probVector, i)
		
		# print("logLikelihood", self.logLikelihood(data))
		return self.logLikelihood(data)
	
	def run(self, data, iterations):
		logLikelihoods = []
		for i in range(iterations):
			logLikelihoods.append(self.OneStep(data))
			if (len(logLikelihoods) > 1):
				if (logLikelihoods[-1] == logLikelihoods[-2]):
					break
		
		return logLikelihoods

	def plotAssignments(self, data, title, imgPath):
		# determine assignments
		assignments = []
		for i in range(data.shape[0]):
			x = np.expand_dims(data[i], axis=1)
			probabilities = []
			for i,model in enumerate(self.gModels):
				probabilities.append(model.getProbability(x) * self.weights[i])
			probabilities = np.array(probabilities)
			assignment = np.argmax(probabilities)
			assignments.append(assignment)
		
		# scatter plot with assignments
		assignments = np.array(assignments)
		plt.scatter(data[:, 0], data[:, 1], c=assignments)
		# plt.legend()
		plt.grid(True)
		plt.title(title)
		if imgPath is not None:
			plt.savefig(imgPath)
		else:
			plt.show()
	
	def logLikelihood(self, data):
		probVectors = []
		for i, model in enumerate(self.gModels):
			probVector = self.EStepSingleModel(data, i) * self.weights[i]
			probVectors.append(probVector)
		probVectors = np.concatenate(probVectors, axis=1)

		logLikelihood = np.sum(np.log(np.sum(probVectors, axis=1)))

		return logLikelihood

def bestKCluster(data, K):
	tryRand = 5
	maxSteps = 1000
	bestLogLikelihood = -np.inf
	bestEM = None

	for _ in range(tryRand):
		em = EM(data.shape[1], K)
		logLikelihoods = em.run(data, maxSteps)
		# print("logLikelihood", logLikelihood)
		if logLikelihoods[-1] > bestLogLikelihood:
			bestLogLikelihood = logLikelihoods[-1]
			bestEM = em
		
		print("logLikelihood", logLikelihoods[-1])
	return bestEM, bestLogLikelihood

def main():
	path = '3D_data_points.txt'
	if len(sys.argv) > 1:
		path = sys.argv[1]
		
	data = loadData(path)
	print(data.shape)
	title = "Original_Data"
	if(data.shape[1] > 2):
		data = pca(data)
		title = "PCA_Data"
	else:
		data = (data - data.mean()) / data.std()
		data = data.values

	imgPrefix = path.split('.')[0]

	scatterPlot(data, title, imgPrefix + title + ".png")

	print("========== EM Clustering ==========")

	kVsLogLikelihood = []

	for K in range(3, 9):
		em, logLikelihood = bestKCluster(data, K)
		kVsLogLikelihood.append((K, logLikelihood))
		print("K = ", K, "logLikelihood", logLikelihood)
		em.plotAssignments(data, f"k = {K}", f"{imgPrefix}_k_{K}.png")
	
	kVsLogLikelihood = np.array(kVsLogLikelihood)
	print(kVsLogLikelihood)
	plt.clf()
	plt.scatter(kVsLogLikelihood[:, 0], kVsLogLikelihood[:, 1])
	plt.xlabel("K")
	plt.ylabel("Log Likelihood")
	plt.title("K vs Log Likelihood")

	plt.grid(True)
	# plt.show()
	plt.savefig(f"{imgPrefix}_k_vs_loglikelihood.png")
	


if __name__ == "__main__":
	main()
