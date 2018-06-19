from data_preprocessing import load_lv4_data, load_lv4_cluster_idxs_records

test_student_ids = ['05090432', '06090130', '04350090',
                    '05170304', '06210695', '06360966',
                    '04360455', '06170020', '05140713',
                    '04160816', '06190883',
                    '05370376', '06370022',
                    '05670071', '06670114',
                    '06130974', '05410434',
                    '06130274', '05130861',
                    '06540611', '05541341']

taken_course_names_of_students, clusters = load_lv4_data()
cluster_idxs_records = \
    load_lv4_cluster_idxs_records(taken_course_names_of_students, clusters)


# dep_to_patterns = load_lv2_frequent_patterns('course_selection_patterns_all.txt')
# data, students = load_lv1_data_students()
# department_id_to_name = load_department_id_to_department_name()
#
# for i in range(len(students)):
#     student = students[i]
#     if student['id'] in test_student_ids:
#         department_name = department_id_to_name[student['departmentNo']]
#         course_names = set([course['courseName'] for course in student['takenClassesRecords']])
#         dep_patterns = dep_to_patterns[department_name]
#
#         # the key sorting depends on the intersection between the students and the pattern courses
#         # but if all courses in a pattern is fully contained in student courses
#         # there is no necessary to recommend this pattern so we minus 1000 points of which pattern will get
#         ranked_patterns = sorted(dep_patterns, key=lambda pattern: len(pattern['courses'] & course_names) * 1000 + \
#                                                                    pattern['support'] * 10 - \
#                                                             10000 if pattern['courses'] in course_names else 0, reverse=True)
#         top_3_courses = set([name for pattern in ranked_patterns[:5] for name in pattern['courses']]) - course_names
#         print("Recommendation for {} {} ({}) {}".format(student['id'], student['name'], department_name, top_3_courses))
#
#         test_student_ids.remove(student['id'])
#         if len(test_student_ids) == 0:
#             break

