from data_preprocessing import *

if __name__ == '__main__':
    data, students = load_lv1_data_students()

    for i in range(10):
        _data, _students = load_lv1_data_students()
        assert np.array_equal(data, _data)
        print(i, ' validations passed.')