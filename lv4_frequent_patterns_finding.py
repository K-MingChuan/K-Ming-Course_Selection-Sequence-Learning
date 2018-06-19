

from data_preprocessing import load_lv4_data, translate_lv2_frequent_pattern
from algorithms.fp_growth import find_frequent_itemsets

if __name__ == '__main__':
    """
    input: 
    output:
    """

    taken_course_names_of_students, clusters = load_lv4_data()

    # Convert taken_course_names of each student to cluster index
    cluster_idxs_records = []

    for taken_course_names in taken_course_names_of_students:
        cluster_idxs_record = []
        for course_name in taken_course_names:
            idx = None
            for cluster in clusters:
                if course_name in cluster.course_names:
                    idx = cluster.index
                    break
            if idx is None:
                print('Error: Course {} not found in all clusters.'.format(course_name))
            cluster_idxs_record.append(idx)
        cluster_idxs_records.append(cluster_idxs_record)

    # 套用Fp-growth，研究一下support要多少
    print('Start doing fp-growth...')
    support = 100
    cluster_patterns = list(find_frequent_itemsets(cluster_idxs_records,
                                                   include_support=True, minimum_support=support))

    # 結果已經是sorted所以不用再排序
    print(cluster_patterns)


    # for pattern, support in course_selection_patterns:
    #     if len(pattern) >= 3:
    #         department_name, taken_course_names = translate_lv2_frequent_pattern(pattern)
    #         if department_name:
    #             log = '{},{},{}'.format(department_name, ','.join(taken_course_names), support)
    #             print(log)

