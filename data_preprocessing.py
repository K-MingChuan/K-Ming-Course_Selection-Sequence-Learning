import collections
import json
import numpy as np
from sklearn.cluster import KMeans

MAX_TIME = 7  # student can only have at most 12 semesters (6 grades)
_GRADE_BASE_YEAR = 103

ELECTIVE_TYPE = 4
GENERAL_TYPE = 5


def load_elective_course_mapping_dicts(course_id_to_name=None):
    """
    :return: two dicts: (1) course_id_to_index (2) index_to_course_id
    """
    print("Loading mapping dicts...")
    course_id_to_name = course_id_to_name or remain_taken_courses_from_id_to_name_dict(
                                                load_elective_course_id_to_name())
    course_ids = course_id_to_name.keys()
    return dict((course_num, index) for index, course_num in enumerate(course_ids)), \
           dict((index, course_num) for index, course_num in enumerate(course_ids))


def load_elective_course_id_to_name():
    filenames = ['courses/courses_new_103_1.json', 'courses/courses_new_103_2.json',
                 'courses/courses_new_104_1.json', 'courses/courses_new_104_2.json',
                 'courses/courses_new_105_1.json', 'courses/courses_new_105_2.json',
                 'courses/courses_new_106_1.json', 'courses/courses_new_106_2.json']
    course_id_to_name = {}

    for filename in filenames:
        print('Loading elective courses from {}...'.format(filename))
        with open(filename, 'r', encoding='utf-8') as fr:
            elective_courses = [course for course in json.load(fr)
                                if course['type'] in {ELECTIVE_TYPE, GENERAL_TYPE}]
            for course in elective_courses:
                course_id_to_name[course['courseId']] = course['name']

    print('Elective courses loaded.')
    return course_id_to_name


def remain_taken_courses_from_id_to_name_dict(all_elective_course_id_to_names):
    """
    collect all courses from all classes that all student took
    :return: a dict the key is the course's id, the value is the course's name
    """
    print("Collecting all courses from students...")
    all_taken_course_ids = set()
    with open('students.json', 'r', encoding='utf-8') as fr:
        students = json.load(fr)

    for student in students:
        all_taken_course_ids.update([course['courseId'] for course in student['takenClassesRecords']])

    taken_course_id_to_name = dict(((course_id, name) for course_id, name in all_elective_course_id_to_names.items()
                                    if course_id in all_taken_course_ids))

    print("Courses collected, Count: ", len(taken_course_id_to_name))
    return taken_course_id_to_name


def get_course_time(course):
    semester = course['semester'] - 1  # 1,2  =>  0,1
    grade = course['year'] - _GRADE_BASE_YEAR  # 103~106  =>  0~3
    return grade * 2 + semester


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
    elective_course_id_to_index, _ = load_elective_course_mapping_dicts()
    elective_course_size = len(elective_course_id_to_index)

    # for each student, provide a sequence of course-selection,
    # where the index of this sequence refers to the time of semesters
    # each course-selection is a one-hot vector, with padded the first element is the time, second is department no
    # (num of students, num of semesters,  size of 'time, department' padded with num of courses)
    data = np.zeros((len(students), MAX_TIME, elective_course_size + 2), dtype=np.int32)

    # normalization
    for student_index in range(len(students)):
        student = students[student_index]
        department = student['departmentNo']
        elective_courses = [course for course in student['takenClassesRecords']
                            if course['courseId'] in elective_course_id_to_index]

        # count how many time steps does the student have
        time_set = set()
        for course in elective_courses:
            time = get_course_time(course)
            time_set.add(time)

        n_time_steps = len(time_set)
        padding_time = MAX_TIME - n_time_steps
        time_to_index = dict((time, padding_time + index) for index, time in enumerate((sorted(time_set))))

        # add time, department in front of n of courses
        for time, time_index in time_to_index.items():
            data[student_index][time_index][0] = time
            data[student_index][time_index][1] = department

        for course in elective_courses:
            try:
                time = get_course_time(course)
                time_index = time_to_index[time]
                course_index = elective_course_id_to_index[course['courseId']] + 2
                data[student_index][time_index][course_index] += 1
            except Exception as e:
                print(e)

        if student_index % 100 == 0:
            print(student_index, " students done.")

    print('LV1 Data loaded.')
    return data, np.array(students)


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
    course_id_to_name = remain_taken_couses_by_students()
    _, index_to_course_id = load_elective_course_mapping_dicts(course_id_to_name=course_id_to_name)
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
