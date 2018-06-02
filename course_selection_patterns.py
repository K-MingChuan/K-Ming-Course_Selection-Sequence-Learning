import collections
import json

_MAX_TIME = 7  # student can only have at most 12 semesters (6 grades)
_GRADE_BASE_YEAR = 103


def collect_courses_from_student_taken_classes():
    """
    :return: a dict the key is the course's id, the value is the course's name
    """
    print("Collecting all courses...")
    course_id_to_name = {}
    with open('students.json', 'r', encoding='utf-8') as fr:
        students = json.load(fr)

    for student in students:
        for course in student['takenClassesRecords']:
            course_id_to_name[course['courseId']] = course['courseName']

    print("Courses collected, Count: ", len(course_id_to_name))
    return course_id_to_name


def load_course_mapping_dicts(course_id_to_name=None):
    """
    :return: two dicts: (1) course_id_to_index (2) index_to_course_id
    """
    print("Loading mapping dicts...")
    course_id_to_name = course_id_to_name or collect_courses_from_student_taken_classes()
    course_ids = course_id_to_name.keys()
    return dict((course_num, index) for index, course_num in enumerate(course_ids)), \
           dict((index, course_num) for index, course_num in enumerate(course_ids))


def load_data_lv1():
    with open('students.json', 'r', encoding='utf-8') as fr:
        students = json.load(fr)

    course_id_to_index, _ = load_course_mapping_dicts()
    course_size = len(course_id_to_index)

    data = []

    for student in [student for student in students if not student['transfer']]:
        course_selection_pattern = []

        id = student['id']
        department = int(id[2:4])  # e.g. 03'36'0123 36 is a department number

        for i in range(_MAX_TIME):  # prepare one-hot vector in each time
            course_selection_pattern.append([0] * course_size)

        for course in student['takenClassesRecords']:
            semester = course['semester'] - 1  # 1,2  =>  0,1
            grade = course['year'] - _GRADE_BASE_YEAR  # 103~106  =>  0~3
            time = grade * 2 + semester
            index = course_id_to_index[course['courseId']]
            course_selection_pattern[time][index] += 1

        # add time, department in front of n of courses
        for time in range(_MAX_TIME):
            course_selection_pattern[time] = [time, department] + course_selection_pattern[time]

        data.append(course_selection_pattern)

        if len(data) % 100 == 0:
            print(len(data), " students done.")

    return data


def load_data_lv2():
    pass


def translate_lv1_data(lv1_data):
    """
    :param lv1_data: a list of a sequence of record which each sequence contains many records classes taken in each semester
            decoded in the way of LV1
    :return: department_to_sequences each sequence shows all course's names in each semester
    """
    course_id_to_name = collect_courses_from_student_taken_classes()
    _, index_to_course_id = load_course_mapping_dicts(course_id_to_name=course_id_to_name)
    department_to_sequences = collections.defaultdict(lambda: collections.defaultdict(set))
    count_finished = 0
    for records in lv1_data:
        for record in records:
            time = record[0]
            department = record[1]
            for i in range(len(index_to_course_id)):
                if record[i+2]:
                    course_id = index_to_course_id[i]
                    course_name = course_id_to_name[course_id]
                    department_to_sequences[department][time].add(course_name)
        count_finished += 1
        if count_finished % 50 == 0:
            print(count_finished, " translated.")
    return department_to_sequences


def _lookup_36_department_records():
    department_to_sequences = translate_lv1_data(load_data_lv1())
    sequences = department_to_sequences[36]
    for time in range(_MAX_TIME):
        print("Time: ", time, " Courses: ", sequences[time])


if __name__ == '__main__':
    _lookup_36_department_records()
    print("Done.")
