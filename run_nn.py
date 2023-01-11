import sklearn.metrics as metrics
import neural_network as nn
from data_handler import X_train, X_test, y_train, y_test, n_per_entry

lr = 0.1
epochs = 3500
layers = [n_per_entry, 12, 8, 1]
jvnn = nn.NeuralNetwork(layers)

print(f"Accuracy: {metrics.accuracy_score(y_train, jvnn.predict(X_train))}")
print("--" * 10, "^ACCURACY (UNTRAINED)^" ,"--" * 10)

jvnn.train(input_data=X_train,
           target_data=y_train,
           epochs=epochs,
           lr=lr,
           verbosity="verbose")

print("--" * 10, "^TRAINING^" ,"--" * 10)

print(f"Accuracy: {metrics.accuracy_score(y_train, jvnn.predict(X_train))}")
print("--" * 10, "^ACCURACY (TRAINED)^" ,"--" * 10)

print(f"Accuracy: {metrics.accuracy_score(y_test, jvnn.predict(X_test))}")
print("--" * 10, "^ACCURACY (TRAINED) (VALIDATION)^", "--" * 10)

# Next steps:
# Track training accuracy (epoch)
# Track training loss (epoch)
# Track validation epoch
# Track validation loss
