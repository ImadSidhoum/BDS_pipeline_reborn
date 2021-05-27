#!/usr/bin/env sh

echo "Deploying Production model name=Fashion_MNISTmodel"

# Set enviorment variable for the tracking URL where the Model Registry is

conda env config vars set MLFLOW_TRACKING_URI="sqlite:///mlruns_cnn.db" #

#!/usr/bin/env sh
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

mlflow models serve --model-uri models:/Model_prod/production --no-conda

mlflow models build-docker --model-uri models:/Model_prod/production -n "my-image-name"











