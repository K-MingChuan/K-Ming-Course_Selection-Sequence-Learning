from data_preprocessing import *
import words_preprocessing_utils
from word_vector_utils import *
import jiebas.jieba_utils


def get_word_frequency_vectors(course_texts):
    data, wordset, word_to_index = words_preprocessing_utils.get_word_frequency_vectors(course_texts)
    return data


def get_word_model_vectors(course_texts):
    data = []
    word_model = get_word2vec_model()
    for text in course_texts:
        wv = np.zeros(250)
        for word in jiebas.jieba_utils.cut(text):
            if word in word_model.wv:
                wv += word_model.wv[word]
        data.append(wv)
    return np.array(data)


if __name__ == '__main__':
    courses = load_elective_courses()

    course_texts = []
    for course in courses:
        course_texts.append(course['name'] * 3 + course['classGoal'] + course['outline'] + course['effect'] + \
                           course['departmentGoal'] + course['reference'])

    data = get_word_model_vectors(course_texts)

    print('Kmeans clustering started...')
    kmeans_fit = KMeans(n_clusters=35).fit(data)
    print('Kmeans clustering finished.')

    labels = [str(n) for n in list(kmeans_fit.labels_)]
    cluster_to_courses = collections.defaultdict(list)
    for i in range(len(labels)):
        cluster_to_courses[labels[i]].append(courses[i]['name'])

    for cluster, courses in cluster_to_courses.items():
        print('Cluster : ', cluster, ', Count: ', len(courses), ' Courses: ',  courses)

    with open('course_clusters.json', 'w', encoding='utf-8') as fw:
        json.dump(cluster_to_courses, fw, ensure_ascii=False)
