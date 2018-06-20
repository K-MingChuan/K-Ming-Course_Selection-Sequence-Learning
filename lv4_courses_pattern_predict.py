from data_preprocessing import compute_lv4_frequent_patterns,\
    load_lv4_clusters, lv4_patterns_filter, compute_lv4_diff_sets,\
    load_department_id_to_department_name,\
    lv4_courses_recommendation_service

import random

test_student_ids = ['05090432', '06090130',
                    '05170304', '06210695', '06360966',
                    '04360455', '06170020', '05140713',
                    '04160816', '06190883',
                    '05370376', '06370022',
                    '05670071', '06670114',
                    '06130974', '05410434',
                    '06130274', '05130861',
                    '06540611', '05541341']

cluster_patterns = compute_lv4_frequent_patterns(7000)
print('Total number of patterns: {}'.format(len(cluster_patterns)))

# Choose only pattern with length >= 7
cluster_patterns_ge7 = lv4_patterns_filter(cluster_patterns, min_len=8)

# Compute the difference between student record & patterns for each.
diff_sets = compute_lv4_diff_sets(test_student_ids, cluster_patterns_ge7)

# recommend courses for each student
for student_id in test_student_ids:
    recommendations = lv4_courses_recommendation_service(student_id)

