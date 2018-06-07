from data_preprocessing import *

if __name__ == '__main__':
    department_no = '36'
    data, students = load_lv1_data_students()
    department_to_sequence = translate_lv1_data_into_department_to_sequences(data)
    my_sequence = department_to_sequence[department_no]

    for time in range(MAX_TIME):
        print('Time: ', time, my_sequence[str(time)])
