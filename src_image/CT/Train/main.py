import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///../../mlruns.db")
mlflow.tensorflow.autolog()
# mlflow ui --backend-store-uri sqlite:///mlruns.db

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    directory= "../data",
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode='binary',
    shuffle=True,
)

test_generator = datagen.flow_from_directory(
    directory= "../data",
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode='binary',
    shuffle=True,
)

client = MlflowClient()
for rm in client.list_registered_models():
    if dict(rm)["name"] == "model":
        l = dict(rm)["latest_versions"]
        for elt in l:
            if elt.current_stage == 'Production':
                p = elt.source 

model = tf.keras.models.load_model("../../"+p+"/data/model")

with mlflow.start_run() as run:
    run_uuid = run.info.run_uuid
    print("MLflow Run ID: %s" % run_uuid)
    model.fit(train_generator,
             epochs=1,
             validation_data=test_generator)
    results = model.evaluate(test_generator,batch_size=128)
    mlflow.log_metric("test_loss", results[0])
    mlflow.log_metric("test_accuracy", results[1])
print("test loss, test acc:", results)