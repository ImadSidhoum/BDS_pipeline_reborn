import requests
import pandas as pd
from tools import *
import gensim

p = Preprocessing()

host = '127.0.0.1'
port = '5001'

url = f'http://{host}:{port}/invocations'

headers = {
    'Content-Type': 'application/json',
}

X_train_transformed,X_test_transformed,X_val_transformed,y_train,y_val,y_test = p.preprocessing_text_fit('./data.csv')

X_test_transformed = pd.DataFrame(X_test_transformed.toarray()).iloc[:5]



# test_data is a Pandas dataframe with data for testing the ML model
http_data = X_test_transformed.to_json(orient='split')

r = requests.post(url=url, headers=headers, data=http_data)

print(f'Predictions: {r.text}')