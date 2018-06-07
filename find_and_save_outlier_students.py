from data_preprocessing import *
import numpy as np

if __name__ == '__main__':
    students = load_students(transferred=False)

    with open('cluster_labels.txt', 'r', encoding='utf-8') as fr:
        cluster_labels = [int(c) for c in fr.read().split(',')]

    outlier_students = find_outlier_students(students, cluster_labels)

    translated = []
    for s in outlier_students:
        translated.append(translate_student_course_selection_pattern(s))
    with open('outliers.json', 'w+', encoding='utf-8') as fw:
        json.dump(translated, fw)
    print("Done.")