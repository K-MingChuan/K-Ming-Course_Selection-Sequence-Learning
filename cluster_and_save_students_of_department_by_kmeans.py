from data_preprocessing import *
import collections
import numpy as np

if __name__ == '__main__':
    data, students = load_lv1_data()
    data = np.reshape(data, (len(data), -1))  # flatten sequences

    departments_mask = collections.defaultdict(list)

    for i in range(len(students)):
        student = students[i]
        departments_mask[student['departmentNo']].append(i)

    outlier_students = []

    # find clusters  in each department
    # for department, mask in departments_mask.items()
    dep_data = data[departments_mask['36']]
    dep_students = students[departments_mask['36']]
    dep_cluster_labels = find_clusters_by_kmeans(dep_data, dep_students, n_cluster=40)
    outlier_students.extend(find_outlier_students(dep_students, dep_cluster_labels, n_outlier_cluster=2))

    outlier_patterns = [translate_student_course_selection_pattern(s) for s in outlier_students]

    print(outlier_patterns)
    with open('outlier_patterns.json', 'w+', encoding='utf-8') as fw:
        json.dump(outlier_patterns, fw)
