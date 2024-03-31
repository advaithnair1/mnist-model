import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense

mnist_data = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

#print(len(train_labels))
#print(len(test_labels))

#Scaling the data to a value between 0 and 1
train_images = train_images / 255.
test_images = test_images / 255

train_images = train_images[...,np.newaxis]
test_images = test_iamges[...,np.newaxis]

model = Sequential([
        Conv2D(8, (3, 3), padding = 'SAME', activation = 'relu', input_shape = (28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(), 
        Dense(64, activation = "relu"),
        Dense(64, activation = "relu"),
        Dense(10, activation = "softmax")
    ])
    
print(model.summary())

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'mae'])

history = model.fit(scaled_train_images[...,np.newaxis], train_labels, epochs = 5, batch_size = 256)

frame = pd.DataFrame(history.history)

#Accuracy v Epochs
acc_plot = frame.plot(y="accuracy", title="Accuracy vs Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Accuracy")

#Loss vs epochs graph
acc_plot = frame.plot(y="loss", title = "Loss vs Epochs",legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Loss")

test_loss, test_accuracy, test_mae = model.evaluate(scaled_test_images, test_labels)
print (test_loss)
print(test_accuracy)
print (test_mae)

# Randomly selects an image and generates the model's prediction
num_test_images = scaled_test_images.shape[0]

random_inx = np.random.choice(num_test_images, 4)
random_test_images = scaled_test_images[random_inx, ...]
random_test_labels = test_labels[random_inx, ...]

predictions = model.predict(random_test_images)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.4, wspace=-0.2)

for i, (prediction, image, label) in enumerate(zip(predictions, random_test_images, random_test_labels)):
    axes[i, 0].imshow(np.squeeze(image))
    axes[i, 0].get_xaxis().set_visible(False)
    axes[i, 0].get_yaxis().set_visible(False)
    axes[i, 0].text(10., -1.5, f'Digit {label}')
    axes[i, 1].bar(np.arange(len(prediction)), prediction)
    axes[i, 1].set_xticks(np.arange(len(prediction)))
    axes[i, 1].set_title(f"Categorical distribution. Model prediction: {np.argmax(prediction)}")
    
plt.show()

        
