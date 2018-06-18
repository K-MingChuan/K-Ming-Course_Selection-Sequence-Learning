import jiebas
from jiebas import jieba_utils
import numpy as np

from word_vector_utils import get_word2vec_model, get_word_vector_size


def get_word_frequency_vectors(samples, cut_all=False):
    """
    :param samples: all samples with each sample is a text
    :param cut_all: cut_all to jieba.cut
    :return: (data, wordset, word_to_index), data is the result of getting word frequency vectors (numpy array),
        wordset is a set with all words from samples, word_to_index is a dict whose key is a word and the value is its index
    """
    print('Getting word frequency vectors...')
    wordset = set()
    words_results = []
    for sample in samples:
        words = jieba_utils.cut(sample, cut_all=cut_all)
        words_results.append(words)
        wordset.update(words)

    print('Words count: ', len(wordset))
    word_to_index = dict((word, index) for index, word in enumerate(wordset))
    data = np.zeros((len(samples), len(word_to_index)))
    for i in range(len(words_results)):
        for word in words_results[i]:
            data[i][word_to_index[word]] += 1

    print('Word frequency vectors loaded.')
    return data, wordset, word_to_index


def get_word_model_vectors(course_texts):
    data = []
    word_model = get_word2vec_model()
    for text in course_texts:
        wv = np.zeros(get_word_vector_size())
        for word in jiebas.jieba_utils.cut(text):
            if word in word_model.wv:
                wv += word_model.wv[word]
        data.append(wv)
    return np.array(data)


if __name__ == '__main__':
    test_samples = ['哈哈哈你人真好', '老師你人很好', '妳很靠北欸', '老師老師快理我']
    results, wordset, word_to_index = get_word_frequency_vectors(test_samples)
    print(word_to_index)
    print(results)