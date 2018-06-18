import numpy as np
from keras.models import load_model
from data_preprocessing import *


def test_students_recommendations(model, data, students):
    department_id_to_name = load_department_id_to_department_name()

    test_student_ids = ['04520670', '04360694', '02231090']
    for i in range(len(students)):
        student = students[i]
        if student['id'] in test_student_ids:
            sequence = [seq for seq in data[i]
                        if np.any(seq)]
            x = sequence[0:3]  # use the first three semesters to predict
            predictions = []

            for j in range(3):  # test three times
                padded_x = np.reshape(padding_sequences(x, MAX_TIME), (1, MAX_TIME, -1))
                y = model.predict(padded_x)[0]
                y[0] = round(y[0])
                y[1] = round(y[1])
                # y = np.round(y)
                sorted_prediction = sorted(enumerate(y[2:]), key=lambda t: t[0],
                                           reverse=True)  # get sorted values with indices
                y[[index + 2 for index, val in sorted_prediction[0:10]]] = 1  # see top 10 predicted course's value to 1
                y[[index + 2 for index, val in sorted_prediction[10:]]] = 0  # otherwise 0
                time, department_id, courses = translate_lv1_data(y)
                department_name = department_id_to_name[department_id] \
                    if department_id in department_id_to_name else 'Error dep code'
                print("=" * 20)
                print('Student: {}, Time: {}, Department: {}, Courses: {}'.format(student['name'],
                                                                                  time, department_name, courses))
                predictions.append(y)
                x.append(y)

            test_student_ids.remove(student['id'])
            if len(test_student_ids) == 0:
                break

# ['服務行銷（英）', '競爭策略（英）', '法律概論（英）', '質化研究', '心理學（英）', '華語語意暨語用學', '國際專題（英）', '非傳統安全管理專題研究', '組織行為（英）', '國際貿易實務（英）']
if __name__ == '__main__':
    data, students = load_lv1_data_students()
    sequences, labels = enumerate_sequences_labels(data)
    model = load_model('3500_1lstm_adam_e100_b256.hdf5')
    model.summary()
    test_students_recommendations(model, data, students)