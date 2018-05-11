from graphviz import Digraph

from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
from keras.utils.vis_utils import plot_model

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = pd.read_csv("C:\pythonhackethon\pedalshare.csv")
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:3].values
Y = dataset.iloc[:,3].values

# create model
model = Sequential()
model.add(Dense(3, input_dim=3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=25, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

