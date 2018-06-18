from data_preprocessing import *
import words_preprocessing_utils
from word_vector_utils import *
import jiebas.jieba_utils


def get_word_frequency_vectors(course_texts):
    data, wordset, word_to_index = words_preprocessing_utils.get_word_frequency_vectors(course_texts)
    return data


def get_word_model_vectors(course_texts):
    return words_preprocessing_utils.get_word_model_vectors(course_texts)


if __name__ == '__main__':
    courses = load_elective_courses()

    course_texts = []
    for course in courses:
        course_texts.append(course['name'] * 3 + course['classGoal'] + course['outline'] + course['effect'] + \
                           course['departmentGoal'] + course['reference'])

    data = get_word_model_vectors(course_texts)

    n_clusters = 50
    print('Kmeans clustering started...')
    kmeans_fit = KMeans(n_clusters=n_clusters).fit(data)
    print('Kmeans clustering finished.')

    labels = [str(n) for n in list(kmeans_fit.labels_)]
    cluster_to_courses = collections.defaultdict(list)
    for i in range(len(labels)):
        cluster_to_courses[labels[i]].append(courses[i]['name'])

    for cluster, courses in cluster_to_courses.items():
        print('Cluster : ', cluster, ', Count: ', len(courses), ' Courses: ',  courses)

    with open('course_clusters_' + str(n_clusters) + '.txt', 'w', encoding='utf-8') as fw:
        for cluster, courses in cluster_to_courses.items():
            fw.write(','.join(courses) + '\n')
