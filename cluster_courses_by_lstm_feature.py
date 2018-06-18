
from data_preprocessing import *
from word_vector_utils import *
from keras.layers import *
from keras.models import *
import numpy as np
courses = load_elective_courses()
course_texts = []

for course in courses:
    course_texts.append(course['name'] * 3 + course['classGoal'] + course['outline'] + course['effect'] + \
                        course['departmentGoal'] + course['reference'])

data = []
word2vec_model = get_word2vec_model()
for text in course_texts:
    word_vectors = []
    if text in word2vec_model.wv:
        word_vectors.append(word2vec_model.wv[text])
    data.append(word_vectors)

n_time_step = 150
data = pad_sequences(data, maxlen=n_time_step)

print(data.shape)
model = Sequential()
model.add(LSTM(128, input_shape=(n_time_step, get_word_vector_size())))
model.add(Activation('sigmoid'))
model.add(Dense(100))

course_fetures = []

for datus in data:
    x = np.reshape(datus, (1, n_time_step, get_word_vector_size()))
    y = model.predict(x)[0]
    course_fetures.append(y)

find_clusters_by_kmeans(course_fetures)