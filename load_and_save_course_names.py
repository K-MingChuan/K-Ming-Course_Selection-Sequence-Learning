from data_preprocessing import *

if __name__ == '__main__':
    courses = load_elective_courses()
    course_names = [course['name'] for course in courses]
    with open('taken_course_names.txt', 'w+', encoding='utf-8') as fw:
        for name in course_names:
            fw.write(name + '\n')