

from data_preprocessing import load_lv4_data, translate_lv2_frequent_pattern
from algorithms.fp_growth import find_frequent_itemsets

if __name__ == '__main__':
    """
    input: student_id
    output: courses_recommendation_list, each course is an object
    """

    # 先讀取: 1. 所有學生的修課名稱資料 2. clusters 的資料
    student_records, courses_clusters = load_lv4_data()

    # 遍歷每個學生的 record 再核對修的課出現在那些群，轉換後存成 list
    student_clusters_records = []

    for student_record in student_records:
        pass

    # 套用Fp-growth，研究一下support要多少
    support = None
    cluster_patterns = list(find_frequent_itemsets(student_clusters_records,
                                                    include_support=True, minimum_support=support))

    # 結果已經是sorted所以不用再排序
    print(cluster_patterns)


    # for pattern, support in course_selection_patterns:
    #     if len(pattern) >= 3:
    #         department_name, course_names = translate_lv2_frequent_pattern(pattern)
    #         if department_name:
    #             log = '{},{},{}'.format(department_name, ','.join(course_names), support)
    #             print(log)

