

from data_preprocessing import load_lv4_data #, translate_lv2_frequent_pattern
from algorithms.fp_growth import find_frequent_itemsets

if __name__ == '__main__':
    """
    input: student_id
    output: courses_recommendation_list, each course is an object
    """
    data = load_lv4_data()

    for d in data:
        print(d)

    # course_selection_patterns = find_frequent_itemsets(data, include_support=True, minimum_support=7)
    # for pattern, support in course_selection_patterns:
    #     if len(pattern) >= 3:
    #         department_name, course_names = translate_lv2_frequent_pattern(pattern)
    #         if department_name:
    #             log = '{},{},{}'.format(department_name, ','.join(course_names), support)
    #             print(log)

