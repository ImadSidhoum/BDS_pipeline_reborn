import os
import numpy as np
import tensorflow as tf
import mlflow
import pickle,sys
from tools import *
import gensim

p = Preprocessing()

X_train_transformed,X_test_transformed,X_val_transformed,y_train,y_val,y_test = p.preprocessing_text_fit('./data.csv')
print(X_train_transformed.toarray().shape)
X_train_transformed = pd.DataFrame(X_train_transformed.toarray())
X_test_transformed = pd.DataFrame(X_test_transformed.toarray())
X_val_transformed = pd.DataFrame(X_val_transformed.toarray())


#df = pd.DataFrame({'rep':X_train_transformed.toarray()},index= np.arange(X_train_transformed.toarray().shape[0]))
#print(df.head())
#print(X_train_transformed.toarray())

'''
df['tfidf'] = hero.tfidf(df['sent'])

print(type(X_train_transformed))
X_train_transformed = pd.DataFrame(pd.arrays.SparseArray(X_train_transformed))
print(type(X_train_transformed))
'''
# enable autologging

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1,activation= 'softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("training")
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.tensorflow.autolog()
with mlflow.start_run() as run:
    history = model.fit(X_train_transformed,y_train,
                        epochs=10,
                        validation_data=(X_val_transformed,y_val),
                        verbose=1)

    results = model.evaluate(X_test_transformed,y_test)
    #mlflow.log_artifact(filename)
    mlflow.log_metric("test_loss", results[0])
    mlflow.log_metric("test_accuracy", results[1])

print(model.predict(X_test_transformed))
print("test loss, test acc:", results)