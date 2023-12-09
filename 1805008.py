import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import tabulate
from tabulate import tabulate

class Util:
	def entropyBin(p, n):
		if p == 0 or n == 0:
			return 0
		q = p / (p + n)
		return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

	def remainder(xtr, ytr, col, bins):
		p = int(np.sum(ytr == 1))
		n = int(np.sum(ytr == 0))

		labels = []
		data = None

		# print(col)
		# print(bins, type(bins))

		if bins != None:
			labels = list(range(bins))
			labels = [str(l) for l in labels]
			# print(labels)
			data = pd.cut(xtr[col], bins=bins, labels=labels)

		else:
			labels = list(xtr[col].unique())
			data = xtr[col]

		remainder = 0

		for i in labels:
			currLabels = ytr[data == i]

			pi = int(np.sum(currLabels == 1))
			ni = int(np.sum(currLabels == 0))

			remainder += ((pi + ni) / (p + n)) * Util.entropyBin(pi, ni)

		return remainder

	def featureSelection(xtr, xts, ytr, maxCats, bins, pickTop):
		gains = []
		p = int(np.sum(ytr == 1))
		n = int(np.sum(ytr == 0))
		# print(p)
		# print(n)
		entropy = Util.entropyBin(p, n)

		cols = list(xtr.columns)
		for col in cols:
			# print(col)
			bin = bins
			if len(xtr[col].unique()) <= maxCats:
				bin = None
			# print(bin)
			gain = Util.remainder(xtr, ytr, col, bin)
			gains.append(gain)

		# print(gains)
		sortIs = np.flip(np.argsort(gains))[0:pickTop]
		# print(sortIs)
		pickedCols = [cols[i] for i in sortIs]
		# print(pickedCols)
		# xts[pickedCols]
		return xtr[pickedCols], xts[pickedCols]

	def convertDataTypes(df, numbers):
		for col in df.columns:
			if col in numbers:
				df[col] = pd.to_numeric(df[col], errors="coerce")
			else:
				df[col] = df[col].astype(str).apply(lambda x: x.strip().lower())

		return df

	def renameCols(xtr, xts):
		colNames = {col: f"{i}" for i, col in enumerate(xtr.columns)}
		xtr.rename(columns=colNames, inplace=True)
		xts.rename(columns=colNames, inplace=True)
		return xtr, xts

	def discretisationY(ytr, yts, trueLabel, falseLabel):
		def discY(y):
			y[y == trueLabel] = 1
			y[y == falseLabel] = 0
			return y.astype(float)

		ytr = discY(ytr)
		yts = discY(yts)

		return ytr, yts

	def discretisationX(xtr, xts, cols):
		def discX(df):
			for col in cols:
				oneHotEnc = pd.get_dummies(df[col], prefix=col)
				df.drop(col, axis=1, inplace=True)
				df = pd.concat([df, oneHotEnc], axis=1)

			return df

		xtr = discX(xtr)
		xts = discX(xts)
		return xtr, xts

	def fill(xtr, xts, cols):
		for col in cols:
			mn = xtr[col].mean()
			xtr[col].fillna(mn, inplace=True)
			xts[col].fillna(mn, inplace=True)
		return xtr, xts

	def normalize(xtr, xts, cols):
		for col in cols:
			mean = xtr[col].mean()
			std = xtr[col].std()
			xtr[col] = (xtr[col] - mean) / std
			xts[col] = (xts[col] - mean) / std
		return xtr, xts


class LogisticRegression:
	def __init__(self, x: np.ndarray, y):
		self.x = x
		self.y = y
		self.w = np.random.rand(self.x.shape[1] + 1, 1)
		self.yesThresh = 0.5

	def crossEntropyLoss(yProb, yAct):
		return np.sum(yAct * np.log(yProb) + (1 - yAct) * np.log(1 - yProb)) * (
			-1 / yProb.shape[0]
		)

	def L2regularization(self, lmbd):
		return np.sum(np.abs(self.w**2)) * lmbd

	# def totalCost

	def train(self, lr, lrUpdater, lmbd, errThresh, mxEpochs, pr=False, earlyStop=True):
		N = self.y.shape[0]
		xtr = np.hstack([np.ones((self.x.shape[0], 1)), self.x])
		ytr = np.reshape(self.y, (N, 1))

		errorRate = 1
		epoch = 0

		# losses = []
		errCnt = 10
		errs = np.zeros(errCnt)
		
		while epoch < mxEpochs and errorRate > errThresh:
			z = xtr @ self.w
			z = z.astype(float)
			prob = 1 / (1 + np.exp(-z))

			gradients = np.sum((ytr - prob) * xtr, axis=0) / N
			gradients = gradients.reshape(-1, 1)

			# loss = LogisticRegression.crossEntropyLoss(prob, self.y) + self.L2regularization(lmbd)

			prob[prob > self.yesThresh] = 1
			prob[prob <= self.yesThresh] = 0

			# print(prob.shape)

			errorRate = np.sum(prob != ytr) / N

			if earlyStop:
				# if(np.abs((np.sum(errs) / errCnt) - errorRate) < .000001):
				if (np.sum(errs) / errCnt) == errorRate:
					print("early stopping")
					break

				errs = np.roll(errs, -1)
				errs[errCnt - 1] = errorRate

			if pr:
				pass
				# print(loss)
				print(
					f"error : {errorRate}, acc : { 1 - errorRate} f1 : {f1_score(ytr, prob)} epoch : {epoch}"
				)
			# print(f"f1 : {}")

			self.w = self.w + lr * (gradients - 2 * lmbd * self.w)
			# print(self.w.shape)

			if lrUpdater != None:
				lr = lrUpdater(lr, epoch, errorRate)

			epoch += 1

		# print(gradients)

		# print(prob.shape)

		# print(xtr)

	def predict(self, x, applyThresh):
		ones = np.ones((x.shape[0], 1))
		x1 = np.hstack([ones, x])
		z = x1 @ self.w
		z = z.astype(float)
		prob = 1 / (1 + np.exp(-z))
		if applyThresh:
			prob[prob > self.yesThresh] = 1
			prob[prob <= self.yesThresh] = 0
		return prob


class AdaBoost:
	def __init__(self, x, y, k, sFrac, uniformWeight):
		self.x = x
		self.y = y
		# self.xy = np.hstack([self.x, self.y.reshape(-1, 1)])
		self.k = k
		self.N = self.x.shape[0]
		self.w = np.ones(self.N)
		if uniformWeight:
			self.w *= 1 / self.N
		else:
			self.w[y == 1] *= 0.5 / np.sum(y == 1)
			self.w[y == 0] *= 0.5 / np.sum(y == 0)

		self.samplesFrac = sFrac
		self.numSamples = int(self.N * sFrac)
		self.hs = []
		self.zs = []
		self.yesThresh = 0.5

	def getSamples(self, probs, noSamples):
		index = np.arange(self.N)
		dataIndex = np.random.choice(index, size=noSamples, p=probs)
		return self.x[dataIndex], self.y[dataIndex]

	def oneHypo(self):
		# data = np.random.choice(self.xy, size = self.numSamples, p = self.w)
		# pi = self.y == 1
		# ni = self.y == 0
		# pw = self.w.copy()
		# pw[ni] = 0
		# nw = self.w.copy()
		# nw[pi] = 0

		# pw = pw / np.sum(pw)
		# nw = nw / np.sum(nw)

		# samples = int(self.numSamples / 2)

		# xp,yp = self.getSamples(pw, samples)
		# xn, yn = self.getSamples(nw, samples)

		# xs = np.vstack([xp, xn])
		# ys = np.concatenate([yp, yn])
		xs, ys = self.getSamples(self.w, self.numSamples)
		# print(xs.shape, ys.shape)

		# index = np.arange(self.N)
		# dataIndex = np.random.choice(index, size = self.numSamples, p = self.w)

		# xs = self.x[dataIndex]
		# ys = self.y[dataIndex]

		yp = np.sum(ys == 1)
		yn = np.sum(ys == 0)

		minErrRate = min(yp, yn) / (yp + yn)

		# print(np.sum(ys==1), np.sum(ys == 0))

		# print(f"min Err rate : {minErrRate}")

		lr = 0.1
		lrUpdater = None
		lmbd = 0
		errThresh = .5
		mxEpochs = 1000

		model = LogisticRegression(xs, ys)
		model.train(lr, lrUpdater, lmbd, errThresh, mxEpochs, False, False)

		ypred = model.predict(self.x, True).flatten()

		error = np.sum(self.w[ypred != self.y])

		# print(f"error : {error}")

		if error >= 0.5:
			return False

		# print(self.w.shape)

		self.w[ypred == self.y] *= error / (1 - error)

		# print(np.random.choice(self.w, 10))

		# print(self.w.shape, np.linalg.norm(self.w))

		self.w = self.w / np.sum(self.w)

		# print(np.sum(self.w))

		self.hs.append(model)
		self.zs.append(np.log((1 - error) / error))

		return True

	def train(self):
		i = 0
		while i < self.k:
			# print(i)
			x = self.oneHypo()
			if x:
				i += 1
		self.zs = np.array(self.zs)
		# self.zs = self.zs / np.sum(self.zs)

	def predict(self, x, dummy):
		ones = np.ones((x.shape[0], 1))
		x1 = np.hstack([ones, x])
		pred = np.zeros((x1.shape[0], 1))

		for i in range(self.k):
			currPred = self.hs[i].predict(x, True)
			currPred[currPred == 0] = -1
			pred = pred + currPred * self.zs[i]

		thresh = 0
		pred[pred > thresh] = 1
		pred[pred <= thresh] = 0

		return pred


class Telco:
	def __init__(self, filePath, features):
		self.df = pd.read_csv(filePath)
		self.df.drop("customerID", axis=1, inplace=True)

		self.numbers = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
		self.df = Util.convertDataTypes(self.df, self.numbers)

		self.out = "Churn"
		self.X = self.df.drop(self.out, axis=1)
		self.Y = self.df[[self.out]].copy()

		self.Xtr, self.Xts, self.Ytr, self.Yts = train_test_split(
			self.X, self.Y[self.out].ravel(), test_size=0.2, random_state=13
		)

		self.cont = ["tenure", "MonthlyCharges", "TotalCharges"]

		self.Xtr, self.Xts = Util.fill(self.Xtr, self.Xts, self.cont)

		self.Xtr, self.Xts = Util.normalize(self.Xtr, self.Xts, self.cont)

		self.Ytr, self.Yts = Util.discretisationY(self.Ytr, self.Yts, "yes", "no")

		mxCats = 10
		bins = 5

		pickTop = features

		# if features < 0:
		#   pickTop = self.Xtr.shape[1]

		catCols = list(self.Xtr.columns)
		remove = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
		for col in remove:
			if col in catCols:
				catCols.remove(col)
		self.Xtr, self.Xts = Util.discretisationX(self.Xtr, self.Xts, catCols)

		if features > 0:
			self.Xtr, self.Xts = Util.featureSelection(
				self.Xtr, self.Xts, self.Ytr, mxCats, bins, pickTop
			)

		# self.Xtr, self.Xts = Util.renameCols(self.Xtr, self.Xts)

	def getTrainingData(self):
		return np.array(self.Xtr), np.array(self.Ytr)


class AdultDataset:
	def __init__(self, trainfp, testfp, features):
		self.dfTrain = pd.read_csv(trainfp, header=None)
		self.dfTest = pd.read_csv(testfp, header=None, skiprows=[0])

		self.numbers = [0, 2, 4, 10, 11, 12]
		self.dfTrain = Util.convertDataTypes(self.dfTrain, self.numbers)
		self.dfTest = Util.convertDataTypes(self.dfTest, self.numbers)

		self.out = 14
		self.Xtr = self.dfTrain.drop(self.out, axis=1)
		self.Ytr = self.dfTrain[self.out].ravel()

		self.Xts = self.dfTest.drop(self.out, axis=1)
		self.Yts = self.dfTest[self.out].ravel()

		# self.Xtr, self.Xts, self.Ytr, self.Yts = train_test_split(self.X, self.Y[self.out].ravel(), test_size = .2, random_state = 13)

		self.cont = [0, 2, 10, 11, 12]

		self.Xtr, self.Xts = Util.fill(self.Xtr, self.Xts, self.cont)

		self.Xtr, self.Xts = Util.normalize(self.Xtr, self.Xts, self.cont)
		self.Yts[self.Yts == ">50k."] = ">50k"
		self.Yts[self.Yts == "<=50k."] = "<=50k"

		# self.Yts[self.out] = self.Yts[self.out].apply(lambda x : x[:-1])

		self.Ytr, self.Yts = Util.discretisationY(self.Ytr, self.Yts, ">50k", "<=50k")

		mxCats = 100
		bins = 2

		pickTop = features

		# if features < 0:
		#   pickTop = self.Xtr.shape[1]

		catCols = list(self.Xtr.columns)
		remove = [0, 2, 4, 10, 11, 12]
		for col in remove:
			if col in catCols:
				catCols.remove(col)
		self.Xtr, self.Xts = Util.discretisationX(self.Xtr, self.Xts, catCols)

		for col in self.Xtr.columns:
			if col not in self.Xts.columns:
				self.Xts[col] = 0

		for col in self.Xts.columns:
			if col not in self.Xtr.columns:
				self.Xts.drop(col, axis=1, inplace=True)

		if features > 0:
			self.Xtr, self.Xts = Util.featureSelection(
				self.Xtr, self.Xts, self.Ytr, mxCats, bins, pickTop
			)

		# self.Xtr, self.Xts = Util.renameCols(self.Xtr, self.Xts)

	def getTrainingData(self):
		return np.array(self.Xtr), np.array(self.Ytr)


class CreditCard:
	def __init__(self, trainfp, features):
		self.df = pd.read_csv(trainfp)

		self.numbers = list(self.df.columns)
		self.df = Util.convertDataTypes(self.df, self.numbers)

		self.out = "Class"

		nsamples = 20000
		posdf = self.df[self.df[self.out] == 1]
		negdf = self.df[self.df[self.out] == 0].sample(n=nsamples, random_state=1)

		# self.df = pd.concat([posdf, negdf])
		# self.df = self.df.sample(frac=1, random_state=13)

		posdfX = posdf.drop(self.out, axis=1)
		posdfY = posdf[self.out].ravel()

		negdfX = negdf.drop(self.out, axis=1)
		negdfY = negdf[self.out].ravel()

		pXtr, pXts, pYtr, pYts = train_test_split(
			posdfX, posdfY, test_size=0.2, random_state=21
		)
		nXtr, nXts, nYtr, nYts = train_test_split(
			negdfX, negdfY, test_size=0.2, random_state=21
		)

		self.Xtr = pd.concat([pXtr, nXtr])
		self.Ytr = np.concatenate([pYtr, nYtr])
		
		self.Xts = pd.concat([pXts, nXts])
		self.Yts = np.concatenate([pYts, nYts])


		# self.X = self.df.drop(self.out, axis=1)
		# self.Y = self.df[self.out].ravel()

		# self.Xtr, self.Xts, self.Ytr, self.Yts = train_test_split(
		# 	self.X, self.Y, test_size=0.2, random_state=13
		# )

		self.cont = list(self.Xtr.columns)

		self.Xtr, self.Xts = Util.fill(self.Xtr, self.Xts, self.cont)

		self.Xtr, self.Xts = Util.normalize(self.Xtr, self.Xts, self.cont)

		mxCats = 2
		bins = 5

		pickTop = features

		if features > 0:
			self.Xtr, self.Xts = Util.featureSelection(
				self.Xtr, self.Xts, self.Ytr, mxCats, bins, pickTop
			)

		# pis = self.Ytr == 1

		# catCols = list(self.Xtr.columns)
		# remove = [0, 2,4, 10, 11, 12]
		# for col in remove:
		#   if col in catCols:
		#     catCols.remove(col)

		# self.Xtr, self.Xts = Util.discretisationX(self.Xtr, self.Xts,catCols)

		# for col in self.Xtr.columns:
		#   if col not in self.Xts.columns:
		#     self.Xts[col] = 0

		# for col in self.Xts.columns:
		#   if col not in self.Xtr.columns:
		#     self.Xts.drop(col, axis = 1, inplace = True)

		# self.Xtr, self.Xts = Util.renameCols(self.Xtr, self.Xts)

	def getTrainingData(self):
		return np.array(self.Xtr), np.array(self.Ytr)


def metrics(model, x, y):
	yPred = model.predict(x, True)
	confMat = confusion_matrix(y, yPred)
	tn, fp, fn, tp = confMat.ravel()


	table = [["Accuracy", accuracy_score(y, yPred)],
			 ["Recall", recall_score(y, yPred)],
			 ["Specificity", tn / (tn + fp)],
			 ["Precision", precision_score(y, yPred)],
			 ["False Discovery Rate", fp / (fp + tp)],
			 ["F1 Score", f1_score(y, yPred)]]

	print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

def metricsTrainAndTest(model, xtr, ytr, xts, yts):
	yPredTrain = model.predict(xtr, True)
	confMatTrain = confusion_matrix(ytr, yPredTrain)
	tnTrain, fpTrain, fnTrain, tpTrain = confMatTrain.ravel()

	yPredTest = model.predict(xts, True)
	confMatTest = confusion_matrix(yts, yPredTest)
	tnTest, fpTest, fnTest, tpTest = confMatTest.ravel()

	table = [["Accuracy", accuracy_score(ytr, yPredTrain), accuracy_score(yts, yPredTest)],
			 ["Recall", recall_score(ytr, yPredTrain), recall_score(yts, yPredTest)],
			 ["Specificity", tnTrain / (tnTrain + fpTrain), tnTest / (tnTest + fpTest)],
			 ["Precision", precision_score(ytr, yPredTrain), precision_score(yts, yPredTest)],
			 ["False Discovery Rate", fpTrain / (fpTrain + tpTrain), fpTest / (fpTest + tpTest)],
			 ["F1 Score", f1_score(ytr, yPredTrain), f1_score(yts, yPredTest)]]
	print(tabulate(table, headers=["Metric", "Train", "Test"], tablefmt="double_outline"))
	


def logistic(dataset, config):

	xtr, ytr = dataset.getTrainingData()
	print(xtr.shape, ytr.shape)

	# print(np.sum(dataset.Yts == 1))
	
	# xtr, ytr = dataset.getTrainingData()
	# np.random.seed(13)
	model = LogisticRegression(xtr, ytr)
	model.train(config["lr"], config["lrUpdater"], config["lmbd"], config["errThresh"], config["mxEpochs"], True, False)
	
	metricsTrainAndTest(model, dataset.Xtr, dataset.Ytr, dataset.Xts, dataset.Yts)



def adaBoost(dataset, config):
	fraction = config['fraction']
	trAccs = []
	tsAccs = []
	xtr, ytr = dataset.getTrainingData()

	for k in range(5, 21, 5):
		# fraction = .55
		# np.random.seed(13)
		useUniform = True
		adb = AdaBoost(xtr, ytr, k, fraction, useUniform)
		print(k)
		adb.train()
		trAccs.append(accuracy_score(ytr, adb.predict(xtr, True)))
		tsAccs.append(accuracy_score(dataset.Yts, adb.predict(dataset.Xts, True)))

		# indiAccs = []
		# for h in adb.hs:
		# 	indiAccs.append(accuracy_score(dataset.Ytr, h.predict(dataset.Xtr, True)))
		
		# plt.figure(figsize=(8, 6))
		# plt.plot(list(range(1, k + 1)), indiAccs, marker='o', linestyle='-', label = "Individual Hypothesis Accuracy")
		# plt.plot(list(range(1, k + 1)), [trAccs[-1]] * k, linestyle='--', label = "AdaBoost Accuracy")
		# plt.title('Hypothesis Accuracy (Train)')
		# plt.xlabel('Index')
		# plt.ylabel('Accuracy')
		# plt.ylim(0, 1)  
		# plt.legend()
		# plt.grid(True)
		# plt.show()
		

		# print(f"train acc : {accuracy_score(ytr, adb.predict(xtr, True))}")
		# print(f"test acc : {accuracy_score(dataset.Yts, adb.predict(dataset.Xts, True))}")
	table = [[5, trAccs[0], tsAccs[0]],
			 [10, trAccs[1], tsAccs[1]],
			 [15, trAccs[2], tsAccs[2]],
			 [20, trAccs[3], tsAccs[3]]]
	
	print(tabulate(table, headers=["k", "Train", "Test"], tablefmt="double_outline"))


def main():
	def lrUpdaterTelco(lr, epoch, errorRate):
		lr /= 10 ** (epoch // 1000)
		return lr

	telcoConfig = {
		"lr": 1,
		"lrUpdater": lrUpdaterTelco,
		"lmbd": 0,
		"errThresh": 0,
		"mxEpochs": 1500,
		"fraction" : .55
	}
	
	def lrUpdaterAdult(lr, epoch, err):
		if err > .2:
			return 2
		elif err > .17:
			return .5
		else:
			return 0.1
	
	adultConfig = {
		"lr": 1,
		"lrUpdater": lrUpdaterAdult,
		"lmbd": 0,
		"errThresh": 0,
		"mxEpochs": 1000,
		"fraction" : .55
	}

	creditCardConfig = telcoConfig
	np.random.seed(13)

	config = telcoConfig
	# config = adultConfig
	# config = creditCardConfig

	pathTelco = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
	pathAdultTrain = "./adult/adult.data"
	pathAdultTest = "./adult/adult.test"
	pathCreditCard = "./archive/creditcard.csv"


	dataset = Telco(pathTelco, -1)
	# dataset = AdultDataset(pathAdultTrain, pathAdultTest, -1)
	# dataset = CreditCard(pathCreditCard, -1)

	logistic(dataset, config)
	
	# adaBoost(dataset, config)





if __name__ == "__main__":
	main()
