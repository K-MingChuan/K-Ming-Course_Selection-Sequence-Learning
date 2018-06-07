import collections
import json
import numpy as np
from sklearn.cluster import KMeans

MAX_TIME = 7  # student can only have at most 12 semesters (6 grades)
_GRADE_BASE_YEAR = 103



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
    print('Loading students...')
    with open('students.json', 'r', encoding='utf-8') as fr:
        return [student for student in json.load(fr) if student['tansfer'] == transferred]


def load_lv1_data():
    """
    :return: narrays of (data, all non-transferred students)
    """
    students = load_students(transferred=False)
    course_id_to_index, _ = load_course_mapping_dicts()
    course_size = len(course_id_to_index)

    data = []

    # normalization
    for student in students:
        department = student['departmentNo']

        # for each student, provide a sequence of course-selection,
        # where the index of this sequence refers to the time of semesters
        # each course-selection is a one-hot vector
        course_selection_pattern = []
        for i in range(MAX_TIME):
            course_selection_pattern.append([0] * course_size)

        for course in student['takenClassesRecords']:
            time = get_course_time(course)
            index = course_id_to_index[course['courseId']]
            course_selection_pattern[time][index] += 1

        # add time, department in front of n of courses
        for time in range(MAX_TIME):
            course_selection_pattern[time] = [time, department] + course_selection_pattern[time]


        data.append(course_selection_pattern)

        if len(data) % 100 == 0:
            print(len(data), " students done.")

    print('LV1 Data loaded.')
    return np.array(data), np.array(students)


def load_lv2_data():
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

    for time in range(MAX_TIME):
        print("Time ", time, " : ", time_to_courses[time])
    return {
        'department_id': department_id,
        'department_name': department_name,
        # convert the set of courses into list
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
    """
    find clusters and save labels.
    :param data: normalized data
    :param students: students
    :param n_cluster: number of cluster expected
    :return: a list of cluster labels
    """
    assert len(students) == len(data), "Student size != data size"
    print('Kmeans clustering started...')
    kmeans_fit = KMeans(n_clusters=n_cluster).fit(data)
    print('Kmeans clustering finished.')
    return list(kmeans_fit.labels_)


def find_outlier_students(students, cluster_labels, n_outlier_cluster=2):
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

    # add the least size of n clusters
    for cluster in sorted(cluster_to_students, key=lambda k: len(cluster_to_students[k]))[:n_outlier_cluster]:
        outlier_students.extend(cluster_to_students[cluster])

    # add clusters with same min size
    for students in [students for cluster, students in cluster_to_students.items()
                        if len(students) == min_cluster_size]:
        outlier_students.extend(students)
    return outlier_students
