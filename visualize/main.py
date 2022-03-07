import tensorflow as tf
import matplotlib.pyplot as plt

# load mnist data
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

# plotting
plt.figure()
for c in range(16):
    plt.subplot(4,4,c+1)
    plt.imshow(x_train[c].reshape(28,28), cmap='gray')
plt.show()

# model
input_shape = (28,28,1)
img_input = tf.keras.layers.Input(shape=input_shape)
h = tf.keras.layers.Conv2D(kernel_size=(5,5), filters=16, activation='relu')(img_input)
h = tf.keras.layers.MaxPooling2D((2,2))(h)
h = tf.keras.layers.Conv2D(kernel_size=(5,5), filters=32, activation='relu')(h)
h = tf.keras.layers.MaxPooling2D((2,2))(h)
h = tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu')(h)
h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dense(32, activation='relu')(h)
predictions = tf.keras.layers.Dense(10, activation='softmax')(h)

model = tf.keras.Model(inputs=img_input, outputs=predictions)

model.summary()

# compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, validation_split=0.25, verbose=2)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r-', label='val_loss')
plt.xlabel('epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k-', label='val_accuracy')
plt.xlabel('epoch')
plt.legend()

plt.show()

# model evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(test_loss, test_acc)
model.save('visualization.h5')