import pandas as pd
import ktrain
import sys

args = sys.argv[1:]
path_model = args[0]
path_predict = args[1]
X_predict = pd.read_csv(path_predict) 
predictor = ktrain.load_predictor(path_model)
print(predictor.predict(list(X_predict['Reviews'])))