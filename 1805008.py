import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)


dataPath = './3D_data_points.txt'

data = pd.read_csv(dataPath, header=None)

# print(data.head())

# normalize each column of data
# print(data.mean())
dataNotNormed = data
data = (data - data.mean()) / data.std()


# print(data.head())

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

def scatterPlot(data):
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

components = 2
# dataReduced = pcaUsingSVD(data, components)
dataReduced = pcaUsingCoVarMatrix(data, components)
# dataReduced = pcaUsingLib(dataNotNormed, components)

print(dataReduced.shape)

# scatterPlot(dataReduced)

class MultiVariateGaussian:
    def __init__(self, dimension, mean, covariance):
        self.dimension = dimension
        self.mean = mean
        if mean is None:
            self.mean = np.random.rand(dimension, 1)
        self.covariance = covariance
        if covariance is None:
            self.covariance = np.random.rand(dimension, dimension)
        

    def getProbabilityBatch(self, data):
        assert data.shape[1] == self.dimension

        data = np.expand_dims(data, axis=2) - self.mean
        dataT = np.transpose(data, (0, 2, 1))
        expo = dataT @ np.linalg.inv(self.covariance) @ data

        print("data", data.shape)
        print("dataT", dataT.shape)
        print("expo", expo.shape)

        assert expo.shape == (data.shape[0], 1, 1)
        expo = expo * -0.5
        expo = np.squeeze(expo, axis=2)
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
        prob = prob / np.sqrt(np.linalg.det(self.covariance))
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
    

    def EStepSingleModel(self, data, model):
        # data = data.T
        
        # probabilities = []
        # for j in range(data.shape[0]):
        #     x = np.expand_dims(data[j], axis=1)
        #     probabilities.append(model.getProbability(x))
        
        # probVector = np.array(probabilities).reshape(data.shape[0], 1)
        # return probVector
        return model.getProbabilityBatch(data)
    

    def MStepSingleModel(self, data, probVector, model):
        # data = data.T
        print("probVector", probVector.shape)
        n = np.sum(probVector)
        # mean = probVector * 
        mean = np.multiply(probVector.T, data.T)
        mean = np.expand_dims(np.sum(mean, axis=1), 1)
        mean = mean / n

        print("mean", mean.shape)

        # covariance = np.zeros((self.dimension, self.dimension))
        # for i in range(data.shape[0]):
        #     x = np.expand_dims(data[i], axis=1)
        #     xSubMean = x - mean
        #     covariance += probVector[i] * xSubMean @ xSubMean.T


        data = np.expand_dims(data, axis=2) - mean
        dataT = np.transpose(data, (0, 2, 1))

        covariance = data @ dataT
        covariance = np.multiply(covariance, np.expand_dims(probVector, axis = 2))
        covariance = np.sum(covariance, axis=0) / n

        # print("covariance1", covariance1.shape)
        
        print("covariance", covariance.shape)
        model.updateMeanAndCovariance(mean, covariance)



        pass

    def OneStep(self, data):
        for model in self.gModels:
            probVector = self.EStepSingleModel(data, model)
            self.MStepSingleModel(data, probVector, model)
    
    def run(self, data, iterations):
        for i in range(iterations):
            self.OneStep(data)
        

    def plotAssignments(self, data):
        # determine assignments
        assignments = []
        for i in range(data.shape[0]):
            x = np.expand_dims(data[i], axis=1)
            probabilities = []
            for model in self.gModels:
                probabilities.append(model.getProbability(x))
            probabilities = np.array(probabilities)
            assignment = np.argmax(probabilities)
            assignments.append(assignment)
        
        # scatter plot with assignments
        assignments = np.array(assignments)
        plt.scatter(data[:, 0], data[:, 1], c=assignments)
        plt.show()
    
    def logLikelihood(self, data):
        logLikelihood = 0




        # print("probs",probabilities)
        # probabilities = np.array(probabilities)
        # probabilities /= np.sum(probabilities, axis=0)
        # return probabilities
            
# arr = np.array([[1,2,3],[4,5,6]])
# print(np.sum(arr, axis=1))


em = EM(2, 3)
# em.plotAssignments(dataReduced)
em.run(dataReduced, 6)
em.plotAssignments(dataReduced)
