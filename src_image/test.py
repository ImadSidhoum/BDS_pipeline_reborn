from pprint import pprint
from mlflow.tracking import MlflowClient
import mlflow
# import tensorflow as tf
# pprint(dict(rm), indent=4)

mlflow.set_tracking_uri("sqlite:///mlruns.db")

path=None
client = MlflowClient()
for rm in client.list_registered_models():
    if dict(rm)["name"] == "model":
        l = dict(rm)["latest_versions"]
        for elt in l:
            if elt.current_stage == 'Production':
                path = elt.source 
print(path)

# model = tf.keras.models.load_model(path+"/data/model")
# model.summary()

# import mlflow.pyfunc

# model_name = "model"
# model_version = 1

# model = mlflow.pyfunc.load_model(
#     model_uri=f"models:/{model_name}/{model_version}"
# )

# # model.predict(data)
# print("ok")