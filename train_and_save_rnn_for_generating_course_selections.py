import numpy
from keras.preprocessing.sequence import pad_sequences

from data_preprocessing import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint


x, students = load_lv1_data()
x = pad_sequences(x, maxlen=MAX_TIME, dtype='int32')

print("X's shape: ", x.shape)
'''
y = []
for sequence in x:


y = np.array(y)
print("Y's shape: ", y.shape)

model = Sequential()
model.add(LSTM(256, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1]))
model.compile(loss='mse', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
'''