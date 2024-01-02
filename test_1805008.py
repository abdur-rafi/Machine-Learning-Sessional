import pickle
from train_1805008 import Input,Dense,Softmax,Relu,DropOut, Model, Data, prepareData, acc, macroF1

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
    # predY = model.predict(tsX)
    print(f"Test Accuracy: {acc(model, tsX, tsY)}")
    print(f"Test Macro F1: {macroF1(model, tsX, tsY)}")

if __name__ == '__main__':
    main()

    # print(model)
