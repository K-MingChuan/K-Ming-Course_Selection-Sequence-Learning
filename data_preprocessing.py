import collections
import json
import numpy as np
from sklearn.cluster import KMeans

_MAX_TIME = 7  # student can only have at most 12 semesters (6 grades)
_GRADE_BASE_YEAR = 103

_CLUSTER_LABEL_FILE_NAME = 'cluster_labels.json'


def get_course_time(course):
    semester = course['semester'] - 1  # 1,2  =>  0,1
    grade = course['year'] - _GRADE_BASE_YEAR  # 103~106  =>  0~3
    return grade * 2 + semester


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


def load_students(transferred=False):
    """
    :return: narrays of all students
    """
    with open('students.json', 'r', encoding='utf-8') as fr:
        return [student for student in json.load(fr) if student['tansfer'] == transferred]


def do_course_selection_lv1_statistic_each_department():
    """
    :return: narrays of (data, all non-transferred students)
    """
    students = load_students()
    course_id_to_index, _ = load_course_mapping_dicts()
    course_size = len(course_id_to_index)

    transferred_students = [student for student in students if student['tansfer']]
    print("Transferred students count: ", len(transferred_students))

    students = [student for student in students if not student['tansfer']]
    print("Non-Transferred students count: ", len(students))

    data = []

    # normalization
    for student in students:
        course_selection_pattern = []

        department = student['departmentNo']

        for i in range(_MAX_TIME):  # prepare one-hot vector in each time
            course_selection_pattern.append([0] * course_size)

        for course in student['takenClassesRecords']:
            time = get_course_time(course)
            index = course_id_to_index[course['courseId']]
            course_selection_pattern[time][index] += 1

        # add time, department in front of n of courses
        for time in range(_MAX_TIME):
            course_selection_pattern[time] = [time, department] + course_selection_pattern[time]

        data.append(course_selection_pattern)

        if len(data) % 100 == 0:
            print(len(data), " students done.")

    return np.array(data), np.array(students)


def load_data_lv2():
    pass


def load_department_id_to_department_name():
    with open('department.json', 'r', encoding='utf-8') as fr:
        return json.load(fr)


def translate_student_course_selection_pattern(student):
    id_to_department_name = load_department_id_to_department_name()

    time_to_courses = collections.defaultdict(set)
    department_id = student['departmentNo']
    department_name = id_to_department_name[department_id]

    for course in student['takenClassesRecords']:
        time = get_course_time(course)
        time_to_courses[time].add(course['courseName'])

    print("=" * 10)
    print("=" * 10)
    print("Department: ", department_name, ', student: ', student)

    for time in range(_MAX_TIME):
        print("Time ", time, " : ", time_to_courses[time])
    return {
        'department_id': department_id,
        'department_name': department_name,
        'time_to_courses': dict(((time, list(courses)) for time, courses in time_to_courses.items()))
    }


def translate_lv1_data_into_department_to_sequences(lv1_data):
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
                if record[i + 2]:
                    course_id = index_to_course_id[i]
                    course_name = course_id_to_name[course_id]
                    department_to_sequences[department][time].add(course_name)
        count_finished += 1
        if count_finished % 50 == 0:
            print(count_finished, " translated.")
    return department_to_sequences


def find_clusters_by_kmeans(data, students, n_cluster=60):
    assert len(students) == len(data), "Student size != data size"
    kmeans_fit = KMeans(n_clusters=n_cluster).fit(data)

    cluster_labels = list(kmeans_fit.labels_)
    with open(_CLUSTER_LABEL_FILE_NAME, 'w+', encoding='utf-8') as fw:
        fw.write(','.join([str(l) for l in cluster_labels]))

    return cluster_labels


def find_outlier_students(students, cluster_labels):
    cluster_to_students = collections.defaultdict(list)
    for i in range(len(students)):
        cluster = cluster_labels[i]
        cluster_to_students[cluster].append(students[i])

    min_cluster_size = 1000
    for students in cluster_to_students.values():
        if len(students) < min_cluster_size:
            min_cluster_size = len(students)

    print('Min cluster size: ', min_cluster_size)
    outlier_students = []
    for students in [students for cluster, students in cluster_to_students.items()
                        if len(students) == min_cluster_size]:
        outlier_students.extend(students)
    return outlier_students


def _lookup_36_department_records():
    department_to_sequences = translate_lv1_data_into_department_to_sequences(
        do_course_selection_lv1_statistic_each_department())
    sequences = department_to_sequences[36]
    for time in range(_MAX_TIME):
        print("Time: ", time, " Courses: ", sequences[time])


def load_cluster_labels():
    with open(_CLUSTER_LABEL_FILE_NAME, 'r', encoding='utf-8') as fr:
        return [int(c) for c in fr.read().split(',')]


if __name__ == '__main__':
    students = load_students(transferred=False)
    cluster_labels = load_cluster_labels()
    outlier_students = find_outlier_students(students, cluster_labels)
    translated = []
    for s in outlier_students:
        translated.append(translate_student_course_selection_pattern(s))
    with open('outliers.json', 'w+', encoding='utf-8') as fw:
        json.dump(translated, fw)
    print("Done.")
