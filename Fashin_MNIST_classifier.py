import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Shrinking data numbers
train_images = train_images/255.0
test_images = test_images/255.0


#creating the network
model = keras.Sequential([
    # Flatten input layer
    keras.layers.Flatten(input_shape=(28, 28)),
    # hidden layer
    # Danse layer - each neuron in this layer is connected
    # to every neuron in the next layer
    # relu - rectify linear unit
    keras.layers.Dense(128, activation="relu"),
    # output layer
    # softmax - function to turn the last layer into probability layer
    keras.layers.Dense(10, activation="softmax"),
    ])

# configure the learning process
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# training the model
# epochs - number of iteration over the dataset
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested acc:", test_acc)

prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[0])])
