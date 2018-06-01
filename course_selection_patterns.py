import json

_MAX_TIME = 12  # student can only have at most 12 semesters (6 grades)
_GRADE_BASE_YEAR = 103


def load_all_course_ids_set():
    all_course_ids = set()

    filenames = ['courses_103_1.json', 'courses_103_2.json',
                 'courses_104_1.json', 'courses_104_2.json',
                 'courses_105_1.json', 'courses_105_2.json',
                 'courses_106_1.json', 'courses_106_2.json']

    for filename in filenames:
        print('Scanning ' + filename + '...')
        with open(filename, 'r', encoding='utf-8') as fr:
            courses = json.load(fr)
            all_course_ids.update([course['courseId'] for course in courses])
    return all_course_ids


def load_course_id_to_index_dict():
    return dict((course_num, index) for index, course_num in enumerate(load_all_course_ids_set()))


def load_data_lv1():
    with open('students.json', 'r', encoding='utf-8') as fr:
        students = json.load(fr)

    course_id_to_index = load_course_id_to_index_dict()
    course_size = len(course_id_to_index)

    data = []

    for student in students:
        course_selection_pattern = []

        for i in range(_MAX_TIME):  # prepare one-hot vector in each time
            course_selection_pattern.append([0] * course_size)

        for course in student['takenClassesRecords']:
            semester = course['semester'] - 1  # 1,2  =>  0,1
            grade = course['year'] - _GRADE_BASE_YEAR  # 103~106  =>  0~3
            time = grade * 2 + semester
            index = course_id_to_index[course['courseId']]
            course_selection_pattern[time][index] += 1

            # add time, department in front of n of courses
            course_selection_pattern[time] = [time, student['department']] + course_selection_pattern[time]

        data.append(course_selection_pattern)

    return data


def load_data_lv2():
    pass


if __name__ == '__main__':
    n = load_course_id_to_index_dict()
    print(n)