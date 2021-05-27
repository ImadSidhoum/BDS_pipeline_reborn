#!/usr/bin/env sh

#conda env config vars set MLFLOW_TRACKING_URI="sqlite:///mlruns_cnn.db" # Windows

export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

mlflow models serve --model-uri models:/Model_prod/production --no-conda

mlflow models build-docker --model-uri models:/Model_prod/production -n "my-image-name"











