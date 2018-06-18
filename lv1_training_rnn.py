import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from load_model_and_predict import test_students_recommendations
from data_preprocessing import *

_enumerated_sequences_cache = None


def create_and_compile_model(sequences, weight_file_name=None):
    model = Sequential()
    model.add(LSTM(5000, input_shape=(sequences.shape[1], sequences.shape[2]), dropout=0.2))
    model.add(Activation('sigmoid'))
    model.add(Dense(sequences.shape[2]))
    if weight_file_name:
        print('Loading weights...')
        model = load_model(weight_file_name)
        model.summary()
        model.load_weights(weight_file_name)
    else:
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


def train_and_save_model(model, epoch=20, batch_size=128):
    filepath = "3500_1lstm_adam_e{}_b{}.hdf5".format(epoch, batch_size)
    checkpoint = ModelCheckpoint(filepath)
    callbacks_list = [checkpoint]
    model.fit(sequences, labels, epochs=epoch, batch_size=batch_size, callbacks=callbacks_list)
    return model


if __name__ == '__main__':
    data, students = load_lv1_data_students()
    sequences, labels = enumerate_sequences_labels(data)
    model = create_and_compile_model(sequences)
    model = train_and_save_model(model, epoch=100, batch_size=256)
    test_students_recommendations(model, data, students)

