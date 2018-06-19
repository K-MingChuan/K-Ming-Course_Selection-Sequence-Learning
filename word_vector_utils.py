import logging

import jiebas
import numpy
from gensim.corpora import WikiCorpus
from gensim.models import word2vec


def train_word2vec_model():
    sentences = word2vec.LineSentence("wiki_seg.txt")
    model = word2vec.Word2Vec(sentences, size=get_word_vector_size())
    model.save("word2vec.model")


my_model = None


def get_word2vec_model():
    # from gensim.models.wrappers import FastText

    global my_model
    if not my_model:
        print("Loading word2vec model...")
        # my_model = FastText.load_fasttext_format("D:\開發套件專區\機器學習\wiki.zh")
        my_model = word2vec.Word2Vec.load("word2vec.model")
        print("Loading word2vec model loaded.")
    return my_model


def get_word_vector_size():
    return 250


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = get_word2vec_model()
    print(model.wv.vocab)
    while True:
        try:
            word1, word2 = input('input two words: ').split()
            print(model.wv.similarity(word1, word2))
        except:
            print("Error!")
