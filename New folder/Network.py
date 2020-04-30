import numpy as np
import sys

class MLP:

    def __init__(self, input_layer_dimension, output_layer_dimension, hidden_layers, activation = 'sigmoid'):

        self._input_layer_dimension = input_layer_dimension
        self._output_layer_dimension = output_layer_dimension
        self._hidden_layers = hidden_layers                 # No of hidden layers
        self._output_layer = Layer(output_layer_dimension, hidden_layers, activation = activation)
        self._activation = activation
        self._layers = []
        self._weights = []

    def PrintWeights(self):
        for weight in self._weights:
            print(weight.GetWeightMatrix())
            print("\n")

    def InsertHiddenLayer(self, dimension, activation = 'sigmoid'):
        if (len(self._layers) >= self._hidden_layers):
            sys.stderr.write("Total possible hidden layers is = "+str(self._hidden_layers)+". Can't Add more layers\n")
            sys.exit(0)

        self._layers.append(Layer(dimension, len(self._layers), activation = activation))
        self.__Construct_Joining_Weights()                       # Constructs Weight matrix bridge, between newly added layer with last added layer

        if (len(self._layers) == self._hidden_layers):      # All Hidden Layers added
            self._layers.append(self._output_layer)         # Finally appends the output layer
            self.__Construct_Joining_Weights()

    def __Construct_Joining_Weights(self):
        if (len(self._layers) < 2):
            source_dimension = self._input_layer_dimension + 1      # 1 added for bias term
        else:
            source_dimension = self._layers[len(self._layers) - 2].GetDimension() + 1   # Second last added layer


        sink_dimension = self._layers[len(self._layers) -1].GetDimension()     # Last added Layer

        self._weights.append(Weights(source_dimension, sink_dimension))

    def ForwardPropagation(self, network_input):
        input_vector = np.concatenate((np.array([1]), network_input))        # Adds bias 

        for i in range (0, len(self._layers)):
            self._weights[i].GenerateSignal(input_vector)
            input_signal = self._weights[i].GetSignal()
            self._layers[i].FeedInputSignal(input_signal)
            self._layers[i].TransformInput()
            output_vector = self._layers[i].GetOutput()                     # Output of this layer, is input of next layer
            input_vector = np.concatenate((np.array([1]), output_vector))   # Adds bias

        return output_vector            # Finally returns the output of last layer

    def CalculateError(self, network_output, actual_output):
        return (np.square(network_output - actual_output).sum())/self._output_layer_dimension                      # Sqaure Error 

    def BackwardPropagation(self, network_input, actual_output):
        delta_vector = actual_output
        weight_matrix = None
        for i in range(len(self._layers)-1, -1, -1):
            self._layers[i].CalculateDeltaVector(weight_matrix, delta_vector)
            delta_vector = self._layers[i].GetDeltaVector()
            weight_matrix = self._weights[i].GetWeightMatrix()
        
        input_vector = np.concatenate((np.array([1]), network_input))        # 1 added for bias term
        for i in range(0, len(self._layers)):
            self._weights[i].UpdateWeightMatrix(input_vector, self._layers[i].GetDeltaVector())
            input_vector = np.concatenate((np.array([1]), self._layers[i].GetOutput()))

class Layer:

    def __init__(self, dimension, layer_id, activation = 'sigmoid'):
        self._dimension = dimension
        self._layer_id = layer_id
        self._activation = activation
        self._input_signal = np.zeros(dimension)
        self._output = np.zeros(dimension)
        self._delta_vector = np.zeros(dimension)

    def GetDimension(self):
        return self._dimension

    def __Theta(self, x):
        if (self._activation == 'sigmoid'):
            return 1 / (1 + np.exp(-x))
        if (self._activation == 'linear'):
            return x

    def TransformInput(self):
        self._output = self.__Theta(self._input_signal)        # Sigmoid Function. Calculation Theta(x)

    def __GetTransformGradient(self):
        if (self._activation == 'sigmoid'):
            return self.__Theta(self._input_signal) * (1 - self.__Theta(self._input_signal))    # Theta'(x) = Theta(x) * (1 - Theta(x))
        if (self._activation == 'linear'):
            return np.ones(self._input_signal.shape[0])         # returns 1

    def FeedInputSignal(self, input_signal):
        assert (type(input_signal) == type(self._input_signal)), "Input Signal type mismatch for layer = "+str(self._layer_id)

        assert (input_signal.shape == self._input_signal.shape), "Input Signal dimention mismatch for layer = "+str(self._layer_id)

        self._input_signal = input_signal

    def CalculateDeltaVector(self, input_weight, input_delta):
        if (type(input_weight) == type(None)):      # Calculating delta vector of output layer
            self._delta_vector = 2 * (self._output - input_delta) * self.__GetTransformGradient() # In this case, input_delta is actual output vector
        else:                                       # Calculating delta vector of hidden layers
            self._delta_vector = self.__GetTransformGradient() * np.matmul(input_weight[1:,:], input_delta.T)

    def GetDeltaVector(self):
        return self._delta_vector

    def GetOutput(self):
        return self._output

class Weights:

    def __init__(self, source_dimension, sink_dimension):
        self._source_dimension = source_dimension
        self._sink_dimension = sink_dimension
        self._weight_matrix = np.random.normal(loc = 0, scale = 1, size = (source_dimension , sink_dimension))   # 1 added for bias term
        self._output_signal = np.zeros(sink_dimension)

    def GetWeightMatrix(self):
        return self._weight_matrix

    def GenerateSignal(self, input_vector):
        self._output_signal = np.matmul(input_vector, self._weight_matrix)

    def GetSignal(self):
        return self._output_signal

    def __CalculateGradientMatrix(self, input_vector, delta_vector):
        input_vec = input_vector.reshape((1, input_vector.shape[0]))
        delta_vec = delta_vector.reshape((1, delta_vector.shape[0]))

        return np.matmul(input_vec.T, delta_vec)

    def UpdateWeightMatrix(self, input_vector, delta_vector, learning_rate = 0.01):
        # input_vector is the output generated by the source layer
        # delta_vector is the delta generated by the sink layer
        
        grad_matrix = self.__CalculateGradientMatrix(input_vector, delta_vector)

        self._weight_matrix -= (learning_rate * grad_matrix)

