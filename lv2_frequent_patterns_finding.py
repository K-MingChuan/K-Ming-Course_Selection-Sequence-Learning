
from data_preprocessing import load_lv2_data, translate_lv2_frequent_pattern
from utils.fp_growth import find_frequent_itemsets

if __name__ == '__main__':
    data = load_lv2_data(specified_department_id='36')
    course_selection_patterns = find_frequent_itemsets(data, minimum_support=20)
    for pattern in course_selection_patterns:
        if len(pattern) >= 5 and '36' in pattern:
            department_name, course_names = translate_lv2_frequent_pattern(pattern)
            log = '{},{}'.format(department_name, ','.join(course_names))
            print(log)
            with open('course_selection_patterns4.txt', 'a+', encoding='utf-8') as fw:
                fw.write(log + '\n')

