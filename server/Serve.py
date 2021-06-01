from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow


path=None
mlflow.set_tracking_uri("sqlite:///mlruns.db")
client = MlflowClient()
for rm in client.list_registered_models():
    if dict(rm)["name"] == "model":
        l = dict(rm)["latest_versions"]
        for elt in l:
            if elt.current_stage == 'Production':
                path = elt.source 
print(path)

# Initialisation
app = FastAPI()

# requets
class Item(BaseModel):
    data: list


@app.get('/')
async def index():
    return "Server Up"


@app.post('/predict')
async def predict(item:Item):
    path=None
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    client = MlflowClient()
    for rm in client.list_registered_models():
        if dict(rm)["name"] == "model1":
            l = dict(rm)["latest_versions"]
            for elt in l:
                if elt.current_stage == 'Production':
                    path = elt.source 
    if path:
        model = tf.keras.models.load_model(path+"/data/model")
        print(model.summary())
        data = np.array(item.data)
        return model.predict(data).tolist()
    
    else:
        return 'model not found'