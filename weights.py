import tensorflow as tf
from keras.models import load_model
import numpy as np

a = []

model = load_model('./model_new.h5')
for i in range(27):
	a.append(model.layers[i].get_weights())
np.save('driving_weights.npy',a)

