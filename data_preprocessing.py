import collections
import json
import operator
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.cluster import KMeans
from algorithms.fp_growth import find_frequent_itemsets

MAX_TIME = 7  # student can only have at most 12 semesters (6 grades)
_GRADE_BASE_YEAR = 103

ELECTIVE_TYPE = 4
GENERAL_TYPE = 5

_elective_course_id_to_name_cache = None
_elective_course_id_to_index_cache = None
_elective_course_index_to_id_cache = None

_cluster_idxs_records_cache = None
_elective_course_names = None

def load_students_by_id(list_of_id):
    """
    :param list_of_id: e.g. ['03361234', '04361234'] a list of students' id
    :return: students of the ids owner
    """
    students = load_students()
    return [s for s in students if s['id'] in list_of_id]


def load_students_course_set(list_of_id):
    """
    :param list_of_id: e.g. ['03361234', '04361234'] a list of students' id
    :return: a dict its key is the student's id with the value his course set
    """
    students = load_students_by_id(list_of_id)
    id_to_courses = {}
    for student in students:
        id_to_courses[student['id']] = student['takenClassesRecords']
    return id_to_courses


def load_elective_course_mapping_id_index(course_id_to_name=None):
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
                if course['type'] in {ELECTIVE_TYPE, GENERAL_TYPE}:
                    if course['name'] not in course_names:
                        courses.append(course)
                        course_names.add(course['name'])

    print('Elective courses loaded, count: ', len(courses))
    return courses


def load_elective_course_mapping_name_index():
    with open('course_names.txt', 'r', encoding='utf-8') as fr:
        course_names = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
        return dict((name, index) for index, name in enumerate(course_names)),\
                dict((index, name) for index, name in enumerate(course_names))


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


def load_students(transferred=False, department_id=None):
    """
    :return: narrays of all students
    """
    print('Loading students...')
    with open('students.json', 'r', encoding='utf-8') as fr:
        return [student for student in json.load(fr) if student['tansfer'] == transferred
                and not department_id or student['departmentNo'] == department_id]


def load_lv1_data_students():
    """
    :return: narrays of (data, all non-transferred students)
    """

    students = load_students(transferred=False)
    elective_course_name_to_index, _ = load_elective_course_mapping_name_index()
    elective_course_size = len(elective_course_name_to_index)
    print('Index of 程式設計 (一): ', elective_course_name_to_index['程式設計 (一)'])

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
                            if course['courseName'] in elective_course_name_to_index]

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
                course_index = elective_course_name_to_index[course['courseName']] + 2
                data[student_index][time_index][course_index] += 1
            except Exception as e:
                print(e)

        if student_index % 100 == 0:
            print(student_index, " students done.")

    print('LV1 Data loaded.')
    return data, np.array(students)


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
    _, index_to_course_id = load_elective_course_mapping_id_index(course_id_to_name=course_id_to_name)
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
    name_to_index, index_to_name = load_elective_course_mapping_name_index()

    time = data[0]
    department_id = data[1]
    course_names = set()
    for i in range(2, len(data)):
        if data[i]:
            course_name = index_to_name[i - 2]
            course_names.add(course_name)

    return round(time), str(round(department_id)), list(course_names)


def load_lv2_data(specified_department_id=None):
    """
    :return: A list of features of each student used for finding frequent patterns in lv2
    """
    students = load_students(transferred=False)
    elective_course_id_to_index, elective_index_to_course_id = load_elective_course_mapping_id_index()

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
    department_name = department_id_to_name[department_id[0]] if len(department_id) != 0 else None

    course_id_to_name = load_elective_course_id_to_name()
    _, index_to_course_id = load_elective_course_mapping_id_index(course_id_to_name=course_id_to_name)

    course_names = [course_id_to_name[index_to_course_id[index]] for index in frequent_pattern
                    if isinstance(index, int)]
    return department_name, course_names


def load_lv2_frequent_patterns(filename):
    """
    :param filename: the txt file's name of where all patterns save in each line
    :return: a dict its key is the department's name and the value is the department's frequent course-selection patterns
    """
    with open(filename, 'r', encoding='utf-8') as fr:
        dep_to_patterns = collections.defaultdict(list)
        for line in fr.readlines():
            splits = line.strip().split(',')
            dep_name = splits[0]
            courses = splits[1:-1:1]
            support = splits[-1]
            dep_to_patterns[dep_name].append({"support": support, "courses": set(courses)})

    return dep_to_patterns


def load_lv3_data():
    data = []
    students = load_students(transferred=False, department_id='36')
    id_to_name = load_elective_course_id_to_name()
    for student in students:
        student_sequence = []
        time_to_course_names = collections.defaultdict(list)
        for course in student['takenClassesRecords']:
            if course['courseId'] in id_to_name:
                time = get_course_time(course)
                time_to_course_names[time].append(course['courseName'])
        sorted_time_courses = sorted(time_to_course_names.items(), key=operator.itemgetter(0))
        for time, courses in sorted_time_courses:
            student_sequence.append(courses)
        data.append([c for c in student_sequence if len(c) != 0])

        if len(data) % 100 == 0:
            print(len(data), 'Students finished.')
    return [d for d in data if len(d) != 0]

def load_lv4_elective_course_names_set():
    global _elective_course_names
    if _elective_course_names:
        return _elective_course_names
    elective_courses = load_elective_courses()
    elective_course_names = set()
    for elective_course in elective_courses:
        elective_course_names.add(elective_course['name'])
    _elective_course_names = elective_course_names
    return elective_course_names

def student_ids_to_course_names_converter(student_ids, only_elective=False):
    """
    :param student_ids: A list of student ids.
    :param only_elective: No required courses if True
    :return: A list of course names corresponding to each student.
    """
    taken_course_names_of_students = []

    with open('students.json', 'r', encoding='utf-8') as jsonfile:
        student_records = json.load(jsonfile)

    if only_elective:
        elective_course_names = load_lv4_elective_course_names_set()

        for student_id in student_ids:
            for student_record in student_records:
                if not student_id == student_record['id']: continue
                taken_course_names = []
                for course_record in student_record['takenClassesRecords']:
                    if course_record['courseName'] in elective_course_names:
                        taken_course_names.append(course_record['courseName'])
                taken_course_names_of_students.append(taken_course_names)
    else:
        for student_id in student_ids:
            for student_record in student_records:
                if not student_id == student_record['id']: continue
                taken_course_names = []
                for course_record in student_record['takenClassesRecords']:
                    taken_course_names.append(course_record['courseName'])
                taken_course_names_of_students.append(taken_course_names)

    return taken_course_names_of_students

def load_lv4_clusters():
    """
    :return: A dictionary from cluster id to course names.
    """
    with open('lv4_courses_clusters.pattern', 'r', encoding='utf-8') as file:
        lines = []
        for line in file.readlines():
            line = line.strip()
            if len(line) > 0:
                lines.append(line)

    clusters = {}
    for line in lines:
        cluster_id, courses_str = line.split(':')
        courses = set(courses_str.split(','))
        clusters[cluster_id] = courses

    return clusters

def load_lv4_taken_course_names_of_all_students(only_elective=False):
    """
    :param only_elective: Concern only elective courses, no required courses.
    :return: The names of courses taken by each student, as a list.
    """
    with open('students.json', 'r', encoding='utf-8') as jsonfile:
        student_records = json.load(jsonfile)

    taken_course_names_of_all_students = []

    if only_elective:
        elective_course_names = load_lv4_elective_course_names_set()

        for student_record in student_records:
            taken_course_names = []
            for course_record in student_record['takenClassesRecords']:
                if course_record['courseName'] in elective_course_names:
                    taken_course_names.append(course_record['courseName'])
            taken_course_names_of_all_students.append(taken_course_names)
    else:
        for student_record in student_records:
            taken_course_names = []
            for course_record in student_record['takenClassesRecords']:
                taken_course_names.append(course_record['courseName'])
            taken_course_names_of_all_students.append(taken_course_names)

    return taken_course_names_of_all_students

def build_lv4_cluster_idxs_records(taken_course_names_of_students, clusters=None, cached=False):
    """
    :param taken_course_names_of_students: A list of taken course names.
    :param clusters: A dictionary from cluster id to course names
    :return: A list of cluster-indexed records corresponding to each student.
    :cached: Cache the result
    """
    if not clusters:
        clusters = load_lv4_clusters()

    cluster_idxs_records = []

    for taken_course_names in taken_course_names_of_students:
        cluster_idxs_record = []
        for course_name in taken_course_names:
            idx = None
            for cluster_id, course_names in clusters.items():
                if course_name in course_names:
                    idx = cluster_id
                    break
            if idx is None:
                print('Error: Course {} not found in all clusters.'.format(course_name))
            cluster_idxs_record.append(idx)
        cluster_idxs_records.append(cluster_idxs_record)

    if cached:
        global _cluster_idxs_records_cache
        _cluster_idxs_records_cache = cluster_idxs_records.copy()

    return cluster_idxs_records

def compute_lv4_frequent_patterns(support=15000, rebuild=False):
    """
    :param support: Minimup support applied to FP Growth
    :param rebuild: Should be True if students.json
                    or lv4_courses_clusters.pattern is updated
    :return: Course cluster frequent patterns
    """
    global _cluster_idxs_records_cache
    if not rebuild and _cluster_idxs_records_cache:
        print('Found cluster indexes records of students cache.')
    else:
        print('Building cluster indexes records of students...')
        build_lv4_cluster_idxs_records(
            load_lv4_taken_course_names_of_all_students(only_elective=True),
            load_lv4_clusters(), cached=True)

    print('Computing fp-growth with minsup:{} ...'.format(support))
    # If the support is too low (<1000), then it will take about 10 mins.
    cluster_patterns = \
        list(find_frequent_itemsets(_cluster_idxs_records_cache,
                                    include_support=True,
                                    minimum_support=support))
    # We're interested in patterns in descending support, to recommend courses in
    #   the mainstream course-cluster.
    cluster_patterns.sort(reverse=True, key=lambda item: item[1])
    return cluster_patterns

def lv4_patterns_filter(cluster_patterns, min_len=7):
    result = []
    for cluster_pattern in cluster_patterns:
        if len(cluster_pattern[0]) >= min_len:
            # print(cluster_pattern[0])
            result.append(cluster_pattern)

    print('pattern with length at least {}: {}'.format(min_len, len(result)))

    return result

def compute_lv4_diff_sets(student_ids, patterns, only_elective=True):
    taken_course_names_of_students = \
        student_ids_to_course_names_converter(student_ids, only_elective=only_elective)

    # Compute cluster-indexed records of each student
    cluster_indexed_records = build_lv4_cluster_idxs_records(taken_course_names_of_students)

    # 依序將 cluster-indexed record 與 patterns 做比對找出最大交集
    diff_sets = []

    for r_idx, record in enumerate(cluster_indexed_records):
        record = set(record)
        max_match = 0
        max_at = None
        for p_idx, pattern in enumerate(patterns):
            match = len(set(pattern[0]) & record)
            if match > max_match:
                max_match = match
                max_at = p_idx

        student_id = student_ids[r_idx]
        max_pattern = set(patterns[max_at][0])

        diff_set = max_pattern - record
        diff_sets.append(diff_set)

        print('Student id: {}, {} ---len_{}---> {} ---diff---> {}' \
              .format(student_id, record, max_match, max_pattern,
                      diff_set))

    return diff_sets

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