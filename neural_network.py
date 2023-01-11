import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    # return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float128)))

def softmax(array):
    if str(type(array[0])) == "<class 'int'>" or "<class 'float'>" or "<class 'double'>" or "<class 'long'>":
        array = np.array(array) - np.array(array).max()
        return (np.exp(array) / np.sum(np.exp(array))).tolist()

    if str(type(array[0])) == "<class 'list'>":
        softmaxed = []
        for arr in array:
            w = np.exp(arr) / np.sum(np.exp(arr)).tolist()
            softmaxed.append(w.tolist())
        return softmaxed

    else:
        print("softmax type error")

def flatten(training_data):
    datalist = []
    for i in range(len(training_data)):
        datalist.append(training_data[i].reshape(-1))
    return datalist

class NeuralNetwork:

    def __init__(
            self,
            layers
    ):

        self.num_in_nodes = layers[0]
        self.layers = layers
        self.num_out_nodes = layers[-1]
        self.initializeWeights()
        self.model_weights = self.getWeights()
        self.num_layers = len(self.layers)-1
        self.model_activations = [0 for i in range(self.num_layers)]
        self.model_losses = [0 for i in range(self.num_layers)]

    # applying relu function to weights with: / np.sqrt(num_in_nodes/2)


    def initializeWeights(self):
        """
        intializes the weights for the model. going down gives all of the weights for a specific node.
        i.e
        X ==> number of nodes on this layer
        Y ==> number of weights per node on this layer

        returns a list of arrays, each array contains the weights for each layer.
        """
        structure = self.layers
        self.model_weights = []
        for i in range(len(structure) - 1):
            if i > 0:
                self.model_weights.append(
                    (np.random.randn(structure[i], structure[i - 1]) / np.sqrt(self.num_in_nodes / 2)).T)
        #                 self.model_weights.append((1+np.random.randn(structure[i], structure[i-1])).T)

        self.model_weights.append((np.random.randn(structure[-1], structure[-2]) / np.sqrt(self.num_in_nodes / 2)).T)
        #         self.model_weights.append((1+np.random.randn(structure[-1], structure[-2])).T)

        return self.model_weights


    def getWeights(self):
        """
        displays the whole models weights neatly
        """

        return self.model_weights


    def train(self, input_data, target_data, epochs, lr, verbosity):

        for epoch in range(epochs):

            if (epoch + 1) % 500 == 0 and (verbosity == 1 or "verbose"):
                print("Epoch: {}".format(epoch + 1))

            # activations
            for i in range(0, self.num_layers):
                if i == 0:
                    self.model_activations[i] = sigmoid(np.dot(input_data, self.model_weights[i]))
                else:
                    j = i - 1
                    self.model_activations[i] = sigmoid(np.dot(self.model_activations[j], self.model_weights[i]))

            # loss
            for i in range(self.num_layers - 1, -1, -1):
                if i == self.num_layers - 1:
                    self.model_losses[i] = (target_data - self.model_activations[i]) * (self.model_activations[i] * (1 - self.model_activations[i]))
                else:
                    j = i + 1
                    self.model_losses[i] = (self.model_losses[j].dot(self.model_weights[j].T)) * (self.model_activations[i] * (1 - self.model_activations[i]))

            # weights
            for i in range(self.num_layers - 1, -1, -1):
                if i == 0:
                    self.model_weights[i] += input_data.T.dot(self.model_losses[i]) * lr
                else:
                    j = i - 1
                    self.model_weights[i] += self.model_activations[j].T.dot(self.model_losses[i]) * lr

    def predict(self, input_data):

        for i in range(0, len(self.layers)-1):
            if i == 0:
                self.model_activations[i] = sigmoid(np.dot(input_data, self.model_weights[i]))
            else:
                j = i - 1
                self.model_activations[i] = sigmoid(np.dot(self.model_activations[j], self.model_weights[i]))

        predictions = self.model_activations

        for i in range(len(predictions[-1])):
            if predictions[-1][i] > 0.5:
                predictions[-1][i] = 1
            else:
                predictions[-1][i] = 0

        return predictions[-1]
