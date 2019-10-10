# import tensorflow
import tensorflow as tf
print(tf.__version__)

# get fashion mnis data
mnist = tf.keras.datasets.fashion_mnist
# get training and test data
(training_images, training_labels),(test_images,test_labels) = mnist.load_data()

# debug print a training image, and label
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
plt.show()
print(training_labels[0])
print(training_images[0])

# normalizing image instensity value for NN
training_images = training_images/255.0
test_images = test_images / 255.0


# DEFINE Nerual Network model
# 3 layers ( input, hidden, output)
# input layer with the size of the image,
# hidden layer with 128 neurons, output layer as much as the classes
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation= tf.nn.relu),
                                   tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# Flatten for turning images from square dimension to 1 dimension
# Dense add 1 layer containing some neuron
# activation function is needed for every neuron to determine what to do next, which feature that we should emphasize

# BUILD the defined NN model
model.compile(optimizer=tf.train.AdamOptimizer(),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
# TRAIN MODEL
model.fit(training_images, training_labels, epochs=5)
# TEST MODEL
model.evaluate(test_images, test_labels)
