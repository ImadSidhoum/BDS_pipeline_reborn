from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
import numpy as np
from mlflow.tracking import MlflowClient
import mlflow

from starlette_exporter import PrometheusMiddleware, handle_metrics
import prometheus_client as prom
import time

#docker run -it  -p 5001:5001 --mount type=bind,source=$(pwd),target=/app land95/mlflow-server:0.1

# Initialisation
app = FastAPI()
mlflow.set_tracking_uri("sqlite:///mlruns.db")

app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)
histogram_inference_time = prom.Histogram('Inference_time', 'This is inference time')
histogram_class_prediction = prom.Histogram('api_label_pred', 'None')
count_class_prediction = prom.Counter('api_label_pred_number', 'None count')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://grafana:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# requets
class Item(BaseModel):
    data: list


@app.get('/')
async def index():
    return "Server Up"

def count_pred_label(y):
    size_y = y.shape[-1]
    # list_count_label = np.zeros(size_y+1 if size_y==1 else size_y)
    class_y = np.argmax(y, axis=-1)
    for i in range(class_y.shape[0]):
        # list_count_label[class_y[i]] +=1
        histogram_class_prediction.observe(class_y[i])
    # return list_count_label.tolist()

@app.post('/predict/{name}')
async def predict(name, item:Item):
    stage = 'Production'
    # try:
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{stage}")
    data = np.array(item.data)
    start_timer = time.time()
    y = model.predict(data)
    histogram_inference_time.observe(time.time()- start_timer)
    count_class_prediction.inc(data.shape[0])
    return y.tolist()
    # except:
    #     print('404: Model not found.')
    #     return 404

@app.post('/predict/{name}/{stage}')
async def predict(name, stage, item:Item):
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{stage}")
        data = np.array(item.data)
        return model.predict(data).tolist()
    except:
        print('404: Model not found.')
        return 404
