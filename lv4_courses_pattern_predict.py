from data_preprocessing import lv4_courses_recommendation_service,\
    student_ids_to_course_names_converter, lv4_elective_course_names_to_courses_converter
from words_preprocessing_utils import get_word_frequency_vectors

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



# 由學號取得學生修課名稱list
taken_course_names_of_students = \
    student_ids_to_course_names_converter(test_student_ids[:1],
                                          only_elective=True)

print(len(taken_course_names_of_students[0]))
# 再依據課程名稱轉成課程物件
taken_courses_of_students = \
    lv4_elective_course_names_to_courses_converter(taken_course_names_of_students)
# for item in taken_course_names_of_students[0]:
#     print(item)
#
# for item in taken_courses_of_students[0]:
#     print(item)

# 在這之前必須先從物件造出字串
course_texts = []
for course in taken_courses_of_students:
    # 7457
    course_texts.append(course['name']*4 + course['classGoal']*2 + course['outline']*1 + course['effect']*1 + \
                       course['departmentGoal']*0 + course['reference']*0)


data, _, _ = get_word_frequency_vectors(course_texts)




vecs = []



for taken_courses_of_student in taken_courses_of_students:
    pass
# 再利用課程物件丟到斷詞函數去算出vec

# 再從這些vec求出平均

# 再將推薦的群中的每個課算出vec


# 每個課程都去與平均座內基

# 這一群就推薦內基質最大的課

# recommend courses for each student
# for student_id in test_student_ids:
#     recommendations = lv4_courses_recommendation_service(student_id)
#
