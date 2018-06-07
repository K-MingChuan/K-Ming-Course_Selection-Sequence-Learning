from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

from data_preprocessing import *

_enumerated_sequences_cache = None


def create_and_compile_model(sequences, weight_file_name=None):
    model = Sequential()
    model.add(LSTM(256, input_shape=(sequences.shape[1], sequences.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(sequences.shape[2]))
    if weight_file_name:
        print('Loading weights...')
        model.load_weights(weight_file_name)
    model.compile(loss='mse', optimizer='adam')
    return model


def train_and_save_model(model, epoch=20):
    filepath = "model_weights_e{}.hdf5".format(epoch)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(sequences, labels, epochs=epoch, batch_size=1, callbacks=callbacks_list)


if __name__ == '__main__':
    data, students = load_lv1_data_students()
    sequences, labels = enumerate_sequences_labels(data)

    train_and_save_model(create_and_compile_model(sequences), epoch=20)
    train_and_save_model(create_and_compile_model(sequences), epoch=60)
    train_and_save_model(create_and_compile_model(sequences), epoch=90)
    train_and_save_model(create_and_compile_model(sequences), epoch=120)
    train_and_save_model(create_and_compile_model(sequences), epoch=150)
    train_and_save_model(create_and_compile_model(sequences), epoch=200)
    train_and_save_model(create_and_compile_model(sequences), epoch=350)

