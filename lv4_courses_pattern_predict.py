from data_preprocessing import compute_lv4_frequent_patterns,\
    build_lv4_cluster_idxs_records, student_ids_to_course_names_converter,\
    load_lv4_clusters

test_student_ids = ['05090432', '06090130', '04350090',
                    '05170304', '06210695', '06360966',
                    '04360455', '06170020', '05140713',
                    '04160816', '06190883',
                    '05370376', '06370022',
                    '05670071', '06670114',
                    '06130974', '05410434',
                    '06130274', '05130861',
                    '06540611', '05541341']

cluster_patterns = compute_lv4_frequent_patterns(5000)
print('Total number of patterns: {}'.format(len(cluster_patterns)))

cluster_patterns_length_ge7 = []
for cluster_pattern in cluster_patterns:
    if len(cluster_pattern[0]) >= 7:
        print(cluster_pattern)
        cluster_patterns_length_ge7.append(cluster_pattern)

print('pattern with len greater than 7: {}'.format(len(cluster_patterns_length_ge7)))

# 取得所有感興趣的學生各自的所有修課名稱
taken_course_names_of_students = \
    student_ids_to_course_names_converter(test_student_ids, only_elective=True)

# 轉換成 cluster-indexed records
print('-'*40 + 'Students' + '-'*40)

cluster_indexed_records = build_lv4_cluster_idxs_records(taken_course_names_of_students)

# 依序將 cluster-indexed record 與 pattern 做比對找出最大交集
diff_sets = []

for s_idx, cluster_indexed_record in enumerate(cluster_indexed_records):
    record = set(cluster_indexed_record)
    max_match = 0
    max_at = None
    for p_idx, pattern in enumerate(cluster_patterns_length_ge7):
        match = len(set(pattern[0]) & record)
        if match > max_match:
            max_match = match
            max_at = p_idx

    student_id = test_student_ids[s_idx]
    max_pattern = set(cluster_patterns_length_ge7[max_at][0])

    diff_set = max_pattern-record
    diff_sets.append(diff_set)

    print('Student id: {}, {} ---len{}---> {} ---diff---> {}'\
          .format(student_id, record, max_match, max_pattern,
                  diff_set))


# 挑選課程進行推薦

# use cluster id to find course in cluster
clusters = load_lv4_clusters()
## 可能需要將 clusters 重新定義成 dict。



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

