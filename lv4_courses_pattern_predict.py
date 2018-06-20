from data_preprocessing import compute_lv4_frequent_patterns,\
    load_lv4_clusters, lv4_patterns_filter, compute_lv4_diff_sets

import random

test_student_ids = ['05090432', '06090130', '04350090',
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

# Select courses for recommendation
association_dict = {

}

# use cluster id to find course in cluster
clusters = load_lv4_clusters()

for idx, diff_set in enumerate(diff_sets):
    print(test_student_ids[idx], diff_set)
    for cluster_id in diff_set:
        courses = clusters[cluster_id]
        _idx = random.randint(0, len(courses)-1)
        print('└---cluster\'{}\'---> {}'.format(cluster_id, list(courses)[_idx]))

