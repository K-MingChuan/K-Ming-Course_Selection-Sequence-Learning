from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

from data_preprocessing import *

_enumerated_sequences_cache = None


def create_and_compile_model(sequences, weight_file_name=None):
    model = Sequential()
    model.add(LSTM(400, input_shape=(sequences.shape[1], sequences.shape[2])))
    model.add(Dense(sequences.shape[2]))
    if weight_file_name:
        print('Loading weights...')
        model.load_weights(weight_file_name)
    model.compile(loss='mse', optimizer='adam')
    return model


def train_and_save_model(model, epoch=20, batch_size=128):
    filepath = "model_weights_e{}_b{}.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath)
    callbacks_list = [checkpoint]
    model.fit(sequences, labels, epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)


if __name__ == '__main__':
    data, students = load_lv1_data_students()
    sequences, labels = enumerate_sequences_labels(data)

    train_and_save_model(create_and_compile_model(sequences), epoch=10, batch_size=1)

