# import tensorflow as tf
import numpy as np
import requests
import json

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# # Define the text labels
# fashion_mnist_labels = ["T-shirt/top",  # index 0
#                         "Trouser",      # index 1
#                         "Pullover",     # index 2
#                         "Dress",        # index 3
#                         "Coat",         # index 4
#                         "Sandal",       # index 5
#                         "Shirt",        # index 6
#                         "Sneaker",      # index 7
#                         "Bag",          # index 8
#                         "Ankle boot"]   # index 9


# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# (x_train, x_valid) = x_train[500:700], x_train[400:500]
# (y_train, y_valid) = y_train[500:700], y_train[400:500]

# # Reshape input data from (28, 28) to (28, 28, 1)
# w, h = 28, 28
# x_train = x_train.reshape(x_train.shape[0], w, h, 1)
# x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
# x_test = x_test.reshape(x_test.shape[0], w, h, 1)




host = 'localhost' #'0.0.0.0'
port = '5000'

url = f'http://{host}:{port}/invocations'

headers = {
    'Content-Type': 'application/json',
}

# test_data is a Pandas dataframe with data for testing the ML model
# http_data = x_train.to_json(orient='split')

# lists = x_train[10].tolist()
# http_data = json.dumps(lists)

# data = {"inputs": np.ones((1,28,28,1)).tolist()}
data = {"instances": np.zeros((1,28,28,1)).tolist()}
d = json.dumps(data)


r = requests.post(url=url, headers=headers, data=d)

print(f'Predictions: {r.text}')