import numpy as np
import torchvision.datasets as ds
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from typing import List
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

class Input:
	# input shape: (batch_size, #features)
	# output shape: (batch_size, #features)

	def __init__(self, inputFeatures) -> None:
		self.inputShape = (-1, inputFeatures)
		self.outputShape = self.inputShape

	def forward(self, x):
		return x

	def backward(self, gradientLossWRTOutput, _):
		return gradientLossWRTOutput, None

	def clearIOs(self):
		pass



class DropOut:
	def __init__(self,dropProb):
		self.dropProb = dropProb

	def forward(self, x):
		self.mask = np.random.rand(*x.shape) > self.dropProb
		return np.multiply(x, self.mask) # / (1- self.dropProb)

	def backward(self,gradientLossWRTOutput, _):
		return np.multiply(self.mask, gradientLossWRTOutput), None # / (1 - self.dropProb), None

	def initPipeline(self,inputShape,name):
		self.inputShape = inputShape
		self.outputShape = inputShape
		self.name = name

	def clearIOs(self):
		self.maks = None

class Dense:
	# input shape: (batch_size, #features)
	# output shape: (batch_size, #nodes)

	def __init__(self, numNodes) -> None:
		self.numNodes = numNodes

	# input shape: (batch_size, #features)
	def initPipeline(self, inputShape, name):
		inputFeatures = inputShape[1]
		self.features = inputFeatures
		self.name = name
		a = .3
		a = np.sqrt(6 / (inputFeatures + self.numNodes))

		self.weights = np.random.uniform(-a, a, (self.numNodes, inputFeatures))
		self.bias = np.random.uniform(-a, a, (self.numNodes, 1))
		self.outputShape = (-1, self.numNodes)

	# x shape: (batch_size, #features)
	def forward(self, x):
		self.x = x
		self.y = np.dot(self.weights, x.T) + self.bias
		self.y = self.y.T
		return self.y

	# gradientLossWRTOutput shape: (batch_size, #nodes)

	def backward(self, gradientLossWRTOutput,optimizer):

		gradientLossWRTInput = np.dot(gradientLossWRTOutput, self.weights)


		gradientLossWRTWeights = np.dot(gradientLossWRTOutput.T, self.x) / self.x.shape[0]
		gradientLossWRTBias = np.mean(gradientLossWRTOutput, axis = 0, keepdims=True).T

		self.weights = optimizer.update(self.name + '-w', self.weights, gradientLossWRTWeights)
		self.bias = optimizer.update(self.name + '-b', self.bias, gradientLossWRTBias)

		return gradientLossWRTInput, (gradientLossWRTWeights, gradientLossWRTBias)

	def getWeights(self):
		return (self.weights, self.bias)

	def addWeights(self, optimizer):
		optimizer.addWeight(self.name + '-w', self.weights)
		optimizer.addWeight(self.name + '-b', self.bias)

	def clearIOs(self):
		self.x = None
		self.y = None


class Softmax:
	EPSILON = .000001
	MAX = 100

	def __init__(self) -> None:
		pass

	# input shape: (batch_size, #features)
	# output shape: (batch_size, #features)

	def initPipeline(self, inputShape, name):
		self.inputShape = inputShape
		self.outputShape = inputShape
		self.name = name

	def forward(self, x):
		self.x = x
		xc = np.copy(x)
		xc[xc > Softmax.MAX] = Softmax.MAX
		xc[xc < Softmax.EPSILON] = Softmax.EPSILON

		self.y =  np.exp(xc) / np.sum(np.exp(xc), axis=1, keepdims=True)
		return self.y

	# gradientLossWRTOutput shape: (batch_size, #features)

	def backward(self, gradientLossWRTOutput, _):
		n , m = self.y.shape
		gradientOutputWRTInput = np.repeat(self.y, m, axis=0).reshape(n, m, m)
		gradientOutputWRTInput = np.multiply(gradientOutputWRTInput, np.transpose(gradientOutputWRTInput, axes=(0, 2, 1))) * -1

		diagElems = np.reshape(self.y, (n,  m , 1))
		diagElems = diagElems * (1 - diagElems)
		diagElems = np.eye(m) * diagElems
		mask = np.eye(m, dtype=bool)
		mask = np.tile(mask, (n, 1)).reshape(n, m, m)
		gradientOutputWRTInput[mask] = 0
		gradientOutputWRTInput = gradientOutputWRTInput + diagElems

		gradientLossWRTInput = np.matmul(gradientOutputWRTInput, np.expand_dims(gradientLossWRTOutput, -1))
		gradientLossWRTInput = np.squeeze(gradientLossWRTInput)

		return gradientLossWRTInput, None


	def clearIOs(self):
		self.x = None
		self.y = None

class Relu:

	def __init__(self) -> None:
		pass

	def initPipeline(self, inputShape,name):
		self.inputShape = inputShape
		self.outputShape = inputShape
		self.name = name

	def forward(self, x):
		self.x = x
		self.y = np.maximum(x, 0)
		return self.y

	# gradientLossWRTOutput shape: (batch_size, #features)
	def backward(self, gradientLossWRTOutput, _):
		gradientOutputWRTInput = np.where(self.x > 0, 1, 0)
		gradientLossWRTInput = np.multiply(gradientOutputWRTInput, gradientLossWRTOutput)
		return gradientLossWRTInput, None

	def clearIOs(self):
		self.x = None
		self.y = None


class Model:
	EPSILON = .000001
	def __init__(self, *layers) -> None:
		self.nLayers = len(layers)

		for i in range(1, self.nLayers):
			layers[i].initPipeline(layers[i-1].outputShape, f"layer-{i}")

		self.layers = layers

	def forward(self, x):
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def predict(self, x):
		for layer in self.layers:
			if isinstance(layer, DropOut):
				continue
			x = layer.forward(x)
		return x

	def crossEntropyGradient(self, yTrue, yPred):
		ypc = yPred.copy()
		ypc[ypc < Model.EPSILON] = Model.EPSILON
		return - yTrue / ypc

	def __backprop(self,yTrue, yPred, optimizer):
		gradientLossWRTOutput = self.crossEntropyGradient(yTrue, yPred)
		for layer in reversed(self.layers):
			gradientLossWRTOutput, g = layer.backward(gradientLossWRTOutput, optimizer)

	def crossEntropyLoss(self, yPred, yTrue):
		ypc = yPred.copy()
		ypc[ypc < Model.EPSILON] = Model.EPSILON
		return -np.mean(yTrue * np.log(ypc))


	def train(self, x, y, optimizer, epoch, batchSize, perIterCallBack = None, perEpochCallBack = None):

		losses = []

		for layer in self.layers:
			if hasattr(layer, 'addWeights'):
				layer.addWeights(optimizer)


		for j in range(epoch):
			numExa = x.shape[0]

			xy = list(zip(x, y))
			np.random.shuffle(xy)

			x, y = zip(*xy)

			x = np.array(x)
			y = np.array(y)

			numExa = x.shape[0]
			numBatches = numExa // batchSize
			remSamples = numExa % batchSize



			xB = np.array_split(x[:numBatches * batchSize], numBatches)
			yB = np.array_split(y[:numBatches * batchSize], numBatches)

			if remSamples > 0:
				xB.append(x[-remSamples:])
				yB.append(y[-remSamples:])
				numBatches += 1

			for i in range(numBatches):
				xc = xB[i]
				yc = yB[i]

				yPred = self.forward(xc)
				self.__backprop(yc, yPred, optimizer)

				closs = self.crossEntropyLoss(yc, yPred)
				losses.append(closs)
				if perIterCallBack != None:
					perIterCallBack(self)

			optimizer.reset()
			print(f"iteration {j + 1} complete")
			if perEpochCallBack != None:
				perEpochCallBack(self)

		return losses


	def squaredErrorGradient(self, yTrue, yPred):
		return -2 * (yTrue - yPred)

	def clearInputsFromLayers(self):
		for layer in self.layers:
			layer.clearIOs()



class GradientDescent:
	def __init__(self, learningRate) -> None:
		self.learningRate = learningRate

	def addWeight(self, layerName, weight):
		pass

	def update(self, layerName, w, g):
		return w - self.learningRate * g

	def reset(self):
		pass

class Adam:
	def __init__(self, learningRate, beta1 = .9, beta2 = .999, epsilon = 1e-6) -> None:
		self.learningRate = learningRate
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.vs = {}
		self.ss = {}
		self.t = 0

	def addWeight(self, layerName, weight):
		self.vs[layerName] = np.zeros_like(weight)
		self.ss[layerName] = np.zeros_like(weight)

	def update(self, layerName, w, g):
		self.t += 1
		v = self.vs[layerName]
		s = self.ss[layerName]
		v = self.beta1 * v + (1 - self.beta1) * g
		s = self.beta2 * s + (1 - self.beta2) * (g ** 2)
		vc = v / (1 - self.beta1 ** self.t)
		sc = s / (1 - self.beta2 ** self.t)
		self.vs[layerName] = v
		self.ss[layerName] = s

		return w - self.learningRate * vc / (np.sqrt(sc) + self.epsilon)



	def reset(self):
		for key in self.vs:
			self.vs[key] = np.zeros_like(self.vs[key])
			self.ss[key] = np.zeros_like(self.ss[key])
		self.t = 0

class Emnist:
	def __init__(self, path):
		self.train = ds.EMNIST(root = path, split = 'letters', train = True, transform = transforms.ToTensor(), download = True)
		self.test = ds.EMNIST(root = path, split = 'letters', train = False, transform = transforms.ToTensor(), download = True)

	def dataTrain(self):
		return self.train.data.numpy(), self.train.targets.numpy()

	def dataTest(self):
		return self.test.data.numpy(), self.test.targets.numpy()

def eqn(x):
	return 2 * (x[0] ** 2) + 3.5 * x[1] + 7


def oneHotEncode(yTr, yTs):
	encoder = OneHotEncoder(sparse_output=False)
	oneHotYtr = encoder.fit_transform(np.expand_dims(yTr, -1))
	oneHotYts = encoder.transform(np.expand_dims(yTs, -1))

	return oneHotYtr, oneHotYts

def normalize(x):
	return x / 255

def flatten(x):
	return np.reshape(x, (x.shape[0], -1))


def acc(model, x, yTrue):
	yPred = model.predict(x)
	yPred = np.argmax(yPred, axis = 1)
	yTrue = np.argmax(yTrue, axis = 1)
	return accuracy_score(yTrue, yPred)

def plot(loss1, loss2):
	plt.figure(figsize=(8, 5))
	plt.plot(loss1,  linestyle='-', color='green', label='Training')
	plt.plot(loss2, linestyle='-', color='red', label='Validation')
	plt.title('Training')
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

def macroF1(model, x, yTrue):
	yPred = model.predict(x)
	yPred = np.argmax(yPred, axis = 1)
	yTrue = np.argmax(yTrue, axis = 1)
	return f1_score(yTrue, yPred, average='macro')

class Metrics:
  def __init__(self, name):
    self.tL = []
    self.vL = []
    self.tA = []
    self.vA = []
    self.vf1 = []
    self.name = name

def confusionMatrix(model, x, yTrue):
	yPred = model.predict(x)
	yPred = np.argmax(yPred, axis = 1)
	yTrue = np.argmax(yTrue, axis = 1)
	cm = confusion_matrix(yTrue, yPred)
	return cm

def plotConfusionMatrix(cm):
	labels = [chr(i + ord('a')) for i in range(26)]
	plt.figure(figsize=(8, 6))
	# sns.set(font_scale=1.2)
	sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title('Confusion Matrix')

	return plt.gcf()


def getSmallModel(numFeatures, output, dropoutRate = .25):
	model = Model(
		Input(numFeatures),
		Dense(32),
		DropOut(dropoutRate),
		Relu(),
		Dense(output),
		Softmax()
	)
	return model

def getMediumModel(numFeatures, output, dropoutRate = .25):
	model = Model(
		Input(numFeatures),
		Dense(128),
		DropOut(dropoutRate),
		Relu(),
		Dense(64),
		DropOut(dropoutRate),
		Relu(),
		Dense(32),
		DropOut(dropoutRate),
		Relu(),
		Dense(output),
		Softmax()
	)
	return model

def getLargeModel(numFeatures, output, dropoutRate = .25):
	model = Model(
		Input(numFeatures),
		Dense(1024),
		DropOut(dropoutRate),
		Relu(),
		Dense(256),
		DropOut(dropoutRate),
		Relu(),
		Dense(128),
		DropOut(dropoutRate),
		Relu(),
		Dense(64),
		DropOut(dropoutRate),
		Relu(),
		Dense(32),
		DropOut(dropoutRate),
		Relu(),
		Dense(output),
		Softmax()
	)
	return model

class Data:
	def __init__(self):
		self.tX = None
		self.tY = None
		self.vX = None
		self.vY = None
		self.tsX = None
		self.tsY = None

def prepareData():

	rt = Data()
	dataset = Emnist('/content/')
	trX, trY = dataset.dataTrain()
	tsX, tsY = dataset.dataTest()
	trY, rt.tsY = oneHotEncode(trY, tsY)
	trX = normalize(trX)
	tsX = normalize(tsX)
	trX = flatten(trX)
	rt.tsX = flatten(tsX)
	rt.tX, rt.vX, rt.tY, rt.vY = train_test_split(trX, trY, test_size = .15, random_state = 29)
	return rt



def trainModel(model, name, data, optimizer, epoch):

	batchSize = 256
	def perEpochCallBackWrapper(ld : Metrics):

		def perEpochCallBack(model):
			ld.tL.append(model.crossEntropyLoss(data.tY, model.predict(data.tX)))
			ld.vL.append(model.crossEntropyLoss(data.vY, model.predict(data.vX)))

			ld.tA.append(acc(model, data.tX, data.tY))
			ld.vA.append(acc(model, data.vX, data.vY))

			ld.vf1.append(macroF1(model, data.vX, data.vY))

		return perEpochCallBack
	

	metrics = Metrics(name)

	losses = model.train(data.tX, data.tY, optimizer, epoch, batchSize, perEpochCallBack = perEpochCallBackWrapper(metrics))

	print(f"train accuracy : {acc(model, data.tX, data.tY)}")
	print(f"validation accuracy : {acc(model, data.vX, data.vY)}")
	print(f"test accuracy : {acc(model, data.tsX, data.tsY)}")

	return metrics



def trainOneLr(nFeatures,outputs,names,data,lr, epoch):

	dropOutRate = .05
	modelsFunc = [getSmallModel, getMediumModel, getLargeModel]
	metrics = []
	models = []

	for modelFunc, name in zip(modelsFunc, names):

		np.random.seed(37)

		model = modelFunc(nFeatures, outputs, dropOutRate)
		models.append(model)

		optimizer = Adam(lr)
		batchSize = 256

		def perEpochCallBackWrapper(ld : Metrics):
			def perEpochCallBack(model):
				ld.tL.append(model.crossEntropyLoss(data.tY, model.predict(data.tX)))
				ld.vL.append(model.crossEntropyLoss(data.vY, model.predict(data.vX)))

				ld.tA.append(acc(model, data.tX, data.tY))
				ld.vA.append(acc(model, data.vX, data.vY))

				ld.vf1.append(macroF1(model, data.vX, data.vY))

				return perEpochCallBack

		metric = Metrics(name)

		losses = model.train(data.tX, data.tY, optimizer, epoch, batchSize, perEpochCallBack = perEpochCallBackWrapper(metric))

		print(f"train accuracy : {acc(model, data.tX, data.tY)}")
		print(f"validation accuracy : {acc(model, data.vX, data.vY)}")
		print(f"test accuracy : {acc(model, data.tsX, data.tsY)}")

		metrics.append(metric)

	return models, metrics


def plotMetrics(metrics : List[Metrics], models : List[Model], data, lr):
	fig, axes = plt.subplots(3, 3,figsize = (16, 16))
	axes = axes.flatten()

	def plot3(ax,arrs,legends,title):
		n = len(arrs)
		for i in range(n):
			ax.plot(arrs[i], label = legends[i])
			ax.set_title(title)
			ax.grid(True)
			ax.legend()
	tLs = []
	vLs = []
	tAs = []
	vAs = []
	vf1s = []
	legends = []

	for metric in metrics:
		tLs.append(metric.tL)
		vLs.append(metric.vL)
		tAs.append(metric.tA)
		vAs.append(metric.vA)
		vf1s.append(metric.vf1)
		legends.append(metric.name)


	plot3(axes[0], tLs, legends, 'Train Loss')
	plot3(axes[1], vLs, legends, 'Validation Loss')
	plot3(axes[2], tAs, legends, 'Train Accuracy')
	plot3(axes[3], vAs, legends, 'Validation Accuracy')
	plot3(axes[4], vf1s, legends, 'Validation f1')
	fig.delaxes(axes[5])
	cms = [ confusionMatrix(model, data.vX, data.vY) for model in models ]

	labels = [chr(i + ord('A')) for i in range(26)]
	for i in range(len(models)):
		sns.heatmap(cms[i], annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax = axes[6 + i])
		axes[6+i].set_title(f'Vali. Conf. Mat.: {metrics[i].name} model')

	plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0)
	plt.suptitle(f"Learning Rate = {lr}")



def prepareReportImages():
	data = prepareData()
	nFeatures = data.tX.shape[1]
	outputs = data.tY.shape[1]

	names = ['light', 'medium', 'heavy']
	metrics = {}
	lrs = [.001, .0005, .0001, .00005]
	epochs = [10, 10, 10, 10]
	models = {}

	for lr, epoch in zip(lrs, epochs):
		a, b = trainOneLr(nFeatures, outputs, names, data, lr, epoch)
		metrics[lr] = b
		models[lr] = a


def trainBest():
	data = prepareData()
	nFeatures = data.tX.shape[1]
	outputs = data.tY.shape[1]
	np.random.seed(37)
	largeModel = getLargeModel(nFeatures, outputs,.35)
	trainModel(largeModel, 'large', data, Adam(.001), 10)
	trainModel(largeModel, 'large', data, Adam(.0001), 10)
	trainModel(largeModel, 'large', data, Adam(.00001), 10)
	return largeModel

def saveModel(model, path):
	with open(path, 'wb') as f:
		pickle.dump(model, f)
	

def main():
	prepareReportImages()
	model = trainBest()
	saveModel(model, 'model_1805008.pickle')

if __name__ == '__main__':
	main()


