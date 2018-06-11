import collections
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans

MAX_TIME = 7  # student can only have at most 12 semesters (6 grades)
_GRADE_BASE_YEAR = 103

ELECTIVE_TYPE = 4
GENERAL_TYPE = 5

_elective_course_id_to_name_cache = None
_elective_course_id_to_index_cache = None
_elective_course_index_to_id_cache = None

_lv1_data_cache = None
_students_cache = None



def load_elective_course_mapping_dicts(course_id_to_name=None):
    """
    :return: two dicts: (1) course_id_to_index (2) index_to_course_id
    """
    global _elective_course_id_to_index_cache, _elective_course_index_to_id_cache

    if _elective_course_id_to_index_cache and _elective_course_index_to_id_cache:
        return _elective_course_id_to_index_cache, _elective_course_index_to_id_cache

    print("Loading mapping dicts...")
    course_id_to_name = course_id_to_name \
                        or remain_taken_courses_from_id_to_name_dict(load_elective_course_id_to_name())

    course_ids = course_id_to_name.keys()
    course_id_to_index = dict((course_id, index) for index, course_id in enumerate(course_ids))
    index_to_course_id = dict((index, course_id) for index, course_id in enumerate(course_ids))

    if not _elective_course_id_to_index_cache:
        _elective_course_id_to_index_cache = course_id_to_index
    if not _elective_course_index_to_id_cache:
        _elective_course_index_to_id_cache = index_to_course_id

    return _elective_course_id_to_index_cache, _elective_course_index_to_id_cache


def load_elective_courses():
    filenames = ['courses/courses_new_103_1.json', 'courses/courses_new_103_2.json',
                 'courses/courses_new_104_1.json', 'courses/courses_new_104_2.json',
                 'courses/courses_new_105_1.json', 'courses/courses_new_105_2.json',
                 'courses/courses_new_106_1.json', 'courses/courses_new_106_2.json']

    course_names = set()
    courses = []
    for filename in filenames:
        print('Loading elective courses from {}...'.format(filename))
        with open(filename, 'r', encoding='utf-8') as fr:
            for course in json.load(fr):
                if course['type'] in {4, 5}:
                    if course['name'] not in course_names:
                        courses.append(course)
                        course_names.add(course['name'])

            print('Elective courses loaded, count: ', len(courses))
            return courses


def load_elective_course_id_to_name():
    global _elective_course_id_to_name_cache
    if _elective_course_id_to_name_cache:
        return _elective_course_id_to_name_cache

    course_id_to_name = {}

    for course in load_elective_courses():
        course_id_to_name[course['courseId']] = course['name']

    if not _elective_course_id_to_name_cache:
        _elective_course_id_to_name_cache = course_id_to_name

    print('Elective courses to id dict loaded.')
    return _elective_course_id_to_name_cache


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


def load_lv1_data_students():
    """
    :return: narrays of (data, all non-transferred students)
    """
    global _lv1_data_cache, _students_cache
    if _lv1_data_cache and _students_cache:
        return _lv1_data_cache, _students_cache

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
            data[student_index][time_index][1] = int(department)  # normalization

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

    if not _lv1_data_cache:
        _lv1_data_cache = data
    if not _students_cache:
        _students_cache = np.array(students)
    print('LV1 Data loaded.')
    return _lv1_data_cache, _students_cache


def enumerate_sequences_labels(data):
    """
    enumerate all sequences from course-selections
    for example, a sequence with time: 2->3->4->5
    will be enumerated into 3 data, labels:
    [2] -> 3
    [2,3] -> 4
    [2,3,4] -> 5
    :param data: normalized data of course-selections
    :return: enumerated data for training RNN (sequences, labels)
    """
    sequences = []
    labels = []
    progress = 0
    for sequence in data:
        selections = [selection for selection in sequence
                      if np.any(selection)]  # filter off the selections with all zeros
        for i in range(len(selections) - 1):
            enumerated_seqs = [seq.tolist() for seq in selections[:i + 1]]
            padded_seqs = padding_sequences(enumerated_seqs, MAX_TIME)
            sequences.append(padded_seqs)
            labels.append(selections[i + 1])

        progress += 1
        if progress % 50 == 0:
            print(progress, ' sequences have been enumerated.')

    return np.array(sequences), np.array(labels)


def padding_sequences(sequences, max_seq):
    n_padding = max_seq - len(sequences)
    padding_seq = []
    for t in range(n_padding):
        padding_seq.append([0] * len(sequences[0]))
    return padding_seq + sequences


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
    course_id_to_name = remain_taken_courses_from_id_to_name_dict(load_elective_course_id_to_name())
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


def translate_lv1_data(data):
    """
    :param data: feature of lv1 data
    :return: a tuple of (time, department's id, a list courses taken)
    """
    course_id_to_name = load_elective_course_id_to_name()
    course_id_to_index, index_to_course_id = load_elective_course_mapping_dicts(course_id_to_name=course_id_to_name)

    time = data[0]
    department_id = data[1]
    course_names = set()
    for i in range(2, len(data)):
        if data[i]:
            course_id = index_to_course_id[i - 2]
            course_name = course_id_to_name[course_id]
            course_names.add(course_name)

    return time, department_id, list(course_names)


if __name__ == '__main__':
    # test my class
    ids = ['03360296', '03363611', '04362481']
    department_id_to_name = load_department_id_to_department_name()
    data, students = load_lv1_data_students()

    for i in range(len(students)):
        if students[i]['id'] in ids:
            my_data, me = data[i], students[i]
            print('Student: ', me['name'])
            for d in [selection for selection in my_data if np.any(selection)]:
                time, department_id, courses = translate_lv1_data(d)
                print('Time: {}, Department: {}, Courses: {}'.format(time, department_id_to_name[str(department_id)],
                                                                     courses))
            ids.remove(students[i]['id'])
        if len(ids) == 0:
            break


def load_lv2_data(specified_department_id=None):
    """
    :return: A list of features of each student used for finding frequent patterns in lv2
    """
    students = load_students(transferred=False)
    elective_course_id_to_index, elective_index_to_course_id = load_elective_course_mapping_dicts()

    data = []
    for student in students:
        features = set()
        department_id = student['departmentNo']
        if specified_department_id and specified_department_id != department_id:
            continue
        elective_courses = [course for course in student['takenClassesRecords']
                            if course['courseId'] in elective_course_id_to_index]

        # add the department id so that the association btw dep and courses can be detected
        features.add(department_id)

        # enumerate all courses' id into the feature list
        for course in elective_courses:
            course_id = course['courseId']
            features.add(elective_course_id_to_index[course_id])

        data.append(list(features))
    return data


def translate_lv2_frequent_pattern(frequent_pattern):
    """
    :return: department_name, course_names (list)
    """
    assert isinstance(frequent_pattern, list)
    department_id_to_name = load_department_id_to_department_name()
    department_id = [e for e in frequent_pattern
                     if e in department_id_to_name]
    department_name = department_id_to_name[department_id[0]] if len(department_id) != 0 else 'None'

    course_id_to_name = load_elective_course_id_to_name()
    _, index_to_course_id = load_elective_course_mapping_dicts(course_id_to_name=course_id_to_name)

    course_names = [course_id_to_name[index_to_course_id[index]] for index in frequent_pattern
                    if isinstance(index, int)]
    return department_name, course_names
