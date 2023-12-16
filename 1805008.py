import numpy as np

class Input:
    # input shape: (batch_size, #features)
    # output shape: (batch_size, #features)

    def __init__(self, inputFeatures) -> None:
        self.inputShape = (-1, inputFeatures)
        self.outputShape = self.inputShape
    
    def forward(self, x):
        return x
    
    def backward(self, gradientLossWRTOutput, _):
        return gradientLossWRTOutput
    


class Dense:
    # input shape: (batch_size, #features)
    # output shape: (batch_size, #nodes)

    def __init__(self, numNodes) -> None:
        self.numNodes = numNodes
    
    # input shape: (batch_size, #features)
    def initPipeline(self, inputFeatures):
        self.features = inputFeatures

        self.weights = np.random.randn(self.numNodes, inputFeatures)
        self.bias = np.random.randn(self.numNodes)
        self.outputShape = (-1, self.numNodes)

    # x shape: (batch_size, #features)
    def forward(self, x):
        self.x = x
        self.y = np.dot(self.weights, x.T) + self.bias
        return self.y
    
    # gradientLossWRTOutput shape: (batch_size, #nodes)

    def backward(self, gradientLossWRTOutput,learningRate):
        
        gradientLossWRTInput = np.dot(gradientLossWRTOutput, self.weights)
        
        gradientLossWRTWeights = np.dot(gradientLossWRTOutput.T, self.x)
        gradientLossWRTBias = gradientLossWRTOutput.sum(axis=0)
        
        self.weights -= learningRate * gradientLossWRTWeights
        self.bias -= learningRate * gradientLossWRTBias

        return gradientLossWRTInput
        


class Softmax:
    def __init__(self) -> None:
        pass

    # input shape: (batch_size, #features)
    # output shape: (batch_size, #features)

    def initPipeline(self, inputShape):
        self.inputShape = inputShape
        self.outputShape = inputShape

    def forward(self, x):
        self.x = x
        self.y =  np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return self.y
    
    # gradientLossWRTOutput shape: (batch_size, #features)

    def backward(self, gradientLossWRTOutput, _):
        # gradientOutputWRT

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

        jacobian_matrix = np.zeros((n, m, m))

        for i in range(n):
            s = self.forward(self.x[i])
            for j in range(n):
                for k in range(n):
                    jacobian_matrix[i, j, k] = s[j] * (int(j == k) - s[k])


        print(np.isclose(jacobian_matrix, gradientOutputWRTInput).all())

        gradientLossWRTInput = np.multiply(gradientOutputWRTInput, np.expand_dims(gradientLossWRTOutput, -1))

        return gradientLossWRTInput
    

class Relu:
    def __init__(self) -> None:
        pass

    def initPipeline(self, inputShape):
        self.inputShape = inputShape
        self.outputShape = inputShape

    def forward(self, x):
        self.x = x
        self.y = np.maximum(x, 0)
        return self.y

    # gradientLossWRTOutput shape: (batch_size, #features)
    def backward(self, gradientLossWRTOutput, _):
        gradientOutputWRTInput = np.where(self.x > 0, 1, 0)
        gradientLossWRTInput = np.multiply(gradientOutputWRTInput, gradientLossWRTOutput)
        return gradientLossWRTInput
    
    

class Model:
    def __init__(self, layers) -> None:
        for layer in layers:
            if isinstance(layer, Input):
                continue
            layer.initPipeline(layer.inputShape)
        
        self.layers = layers
    
    


# Your initial array of shape (n, m)
original_array = np.array([[1, 2, 3],
                           [4, 5, 6]])

n, m = original_array.shape

# Repeat each row m times
repeated_rows = np.repeat(original_array, m, axis=0)

# Reshape the repeated array to (n, m, m)
result_array = repeated_rows.reshape(n, m, m)

# print(result_array)

# print(np.reshape(original_array, (n,  m , 1)))

diagElems = np.reshape(original_array, (n,  m , 1))
# diagElems = original_array
diagElems = diagElems * (1 - diagElems)
diagElems = np.eye(m) * diagElems
print(diagElems)

mask = np.eye(m, dtype=bool)
mask = np.tile(mask, (n, 1)).reshape(n, m, m)

# result_array = np.fill_diagonal(result_array, diagElems)
# np.fill_diagonal(result_array, 0)
result_array[mask] = 0
result_array = result_array + diagElems

print(result_array)
# print(diagElems)

# print(np.multiply(result_array, np.transpose(result_array, axes=(0, 2, 1))))


# print(np.multiply(result_array, np.reshape(original_array, (n, m, 1))))