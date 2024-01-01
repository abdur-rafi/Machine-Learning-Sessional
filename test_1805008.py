import pickle
from test_1805008 import Data, prepareData, acc, macroF1

def loadModel(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model



def main():
    path = 'model_1805008.pickle'
    model = loadModel(path)
    data = prepareData()
    tsX = data.tsX
    tsY = data.tsY
    predY = model.predict(tsX)
    print(f"Test Accuracy: {acc(tsY, predY)}")
    print(f"Test Macro F1: {macroF1(tsY, predY)}")

if __name__ == '__main__':
    main()

    # print(model)
