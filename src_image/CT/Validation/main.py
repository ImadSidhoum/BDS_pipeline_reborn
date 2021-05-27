import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
import json

datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

path = "data"

train_generator = datagen.flow_from_directory(
    directory= path,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="input",
    shuffle=True,
)

outlier_generator = datagen.flow_from_directory(
    directory= path,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="input",
    shuffle=True,
)

model = VGG16(weights='imagenet',include_top=False, input_shape=(224,224,3))
for layer in model.layers:
    layer.trainable = False

top_model = model.output
output_layer = layers.AveragePooling2D(pool_size=(14, 14), strides=None, padding='same')(top_model)
output_layer = layers.Flatten()(output_layer)

vgg = Model(inputs=model.input, outputs=output_layer)


mean = None
n = 10
for i in range(n):
  pred = vgg.predict(train_generator[i][0]).mean(axis=0)
  if mean is None:
    mean = pred
  else:
    mean = (mean + pred)/2

pred_true = vgg.predict(train_generator[0][0])
loss_true = np.linalg.norm(mean- pred_true, axis=-1)
LIMITE = loss_true.mean() + 2*loss_true.std()

mask = []
for i in range(n):
    pred = vgg.predict(outlier_generator[i][0])
    loss = np.linalg.norm(mean- pred, axis=-1)
    mask.append(loss < LIMITE)

mask= np.array(mask)
mask = mask.flatten()

score = np.sum(mask)/mask.shape[0]
print(mask, score)

dictionary = {"mask": mask.tolist(), "score": score}
print(dictionary)

# Serializing json 
json_object = json.dumps(dictionary)
  
# Writing to sample.json
with open("validation.json", "w") as outfile:
    outfile.write(json_object)