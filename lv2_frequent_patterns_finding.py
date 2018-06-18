
from data_preprocessing import load_lv2_data, translate_lv2_frequent_pattern
from algorithms.fp_growth import find_frequent_itemsets

if __name__ == '__main__':
    data = load_lv2_data(specified_department_id="54")
    course_selection_patterns = find_frequent_itemsets(data, include_support=True, minimum_support=7)
    for pattern, support in course_selection_patterns:
        if len(pattern) >= 3:
            department_name, course_names = translate_lv2_frequent_pattern(pattern)
            if department_name:
                log = '{},{},{}'.format(department_name, ','.join(course_names), support)
                print(log)

