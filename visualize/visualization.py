import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
import os
#from main import CNN
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# adjusting to 0 ~ 1.0
x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape, x_test.shape)

# reshaping
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print(x_train.shape, x_test.shape)

model = load_model("visualization.h5")

layer_name = [layer.name for layer in model.layers]
print(layer_name)
layer_output = [layer.output for layer in model.layers]
print(model.get_layer('conv2d').output)

#visualization_model = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
#CONVOLUTION_NUMBER = 20
#f, axarr = plt.subplot(3,4)

image = x_test[0]
image = image.reshape(-1, 28, 28, 1)
#predict_images = image[0][tf.newaxis, ...]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)
visualization_model_2 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('conv2d').output)
visualization_model_3 = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('max_pooling2d').output)
visualization_model.summary()
first_activation = visualization_model.predict(image)
first_activation2 = visualization_model_2.predict(image)
first_activation3 = visualization_model_3.predict(image)

plt.figure(figsize=(16, 8))
print(first_activation.shape)
print(first_activation.shape[-1])
for i in range(first_activation.shape[-1]):
    plt.subplot(4, 8, i +1 )
    # 눈금 제거. fignum은 같은 피켜에 연속 출력
    plt.axis('off')
    plt.matshow(first_activation[0, :, :, i], cmap='gray', fignum=0)
plt.title("conv2d_1")
plt.tight_layout()
plt.show()
for i in range(first_activation2.shape[-1]):
    plt.subplot(4, 8, i +1 )
    # 눈금 제거. fignum은 같은 피켜에 연속 출력
    plt.axis('off')
    plt.matshow(first_activation2[0, :, :, i], cmap='gray', fignum=0)
plt.title("conv2d")
plt.tight_layout()
plt.show()
for i in range(first_activation3.shape[-1]):

    plt.subplot(4, 8, i +1 )
    # 눈금 제거. fignum은 같은 피켜에 연속 출력
    plt.axis('off')
    plt.matshow(first_activation3[0, :, :, i], cmap='gray', fignum=0)


plt.tight_layout()
plt.show()

#for layer_name, feature_map in zip(layer_name, feature_map): print(
   # f"The shape of the {layer_name} is =======>> {feature_map.shape}")

"""
axarr[0,1].imshow(predict[0, : , :, CONVOLUTION_NUMBER])
plt.show()

print(label[0])
print(predict.argmax(axis=-1))

"""