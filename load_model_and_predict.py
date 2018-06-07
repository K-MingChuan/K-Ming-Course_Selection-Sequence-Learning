from data_preprocessing import *
from train_and_save_rnn_for_generating_course_selections import create_and_compile_model


test_student_ids = ['03360296', '04362481', '03363611']
data, students = load_lv1_data_students()
course_id_to_name = load_elective_course_id_to_name()
course_id_to_index, course_index_to_id = load_elective_course_mapping_dicts()
department_id_to_name = load_department_id_to_department_name()
sequences, labels = enumerate_sequences_labels(data)

model = create_and_compile_model(sequences, weight_file_name='model_weights.hdf5')

for i in range(len(students)):
    student = students[i]
    if student['id'] in test_student_ids:
        _ = data[i]
        sequence = [seq for seq in data[i]
                    if np.any(seq)]
        x = [seq for seq in sequence
             if seq[0] not in [4, 5, 6]]  # we want to predict the sequence in time 4~6, so they shouldn't be included
        predictions = []

        for j in range(3):  # test three times
            padded_x = np.reshape(padding_sequences(x, MAX_TIME), (1, MAX_TIME, -1))
            y = model.predict(padded_x)[0]
            for num in y:
                print(y, ',')
            y = np.round(y)
            time, department_id, courses = translate_lv1_data(y)
            department_name = department_id_to_name[int(department_id)] \
                if department_id in department_id_to_name else 'Error dep code'
            print("="*20)
            print('Student: {}, Time: {}, Department: {}, Courses: {}'.format(student['name'],
                                                                              time, department_name, courses))
            predictions.append(y)
            x.append(y)

        test_student_ids.remove(student['id'])
        if len(test_student_ids) == 0:
            break


