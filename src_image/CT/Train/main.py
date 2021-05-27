import tensorflow as tf
import numpy as np
import sys
import mlflow.tensorflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.tensorflow.autolog()
# mlflow ui --backend-store-uri sqlite:///mlruns.db

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

(x_train, x_valid) = x_train[500:700], x_train[400:500]
(y_train, y_valid) = y_train[500:700], y_train[400:500]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

with mlflow.start_run() as run:
    run_uuid = run.info.run_uuid
    print("MLflow Run ID: %s" % run_uuid)
    model.fit(x_train,
             y_train,
             batch_size=64,
             epochs=1,
             validation_data=(x_valid, y_valid))
    results = model.evaluate(x_test,y_test,batch_size=128)
    mlflow.log_metric("test_loss", results[0])
    mlflow.log_metric("test_accuracy", results[1])
print("test loss, test acc:", results)