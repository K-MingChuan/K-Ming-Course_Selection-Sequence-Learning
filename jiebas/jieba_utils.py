import jieba
import jieba.analyse

puncts = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…\u3000''')

stop_wordset = puncts

print("Preparing jiebas utils...")

jieba_dictionary_name = 'jiebas/dict.txt.big'
jieba.set_dictionary(jieba_dictionary_name)

print("Preparing jiebas utils...")

with open('jiebas/stop_words.txt', 'r', encoding='utf-8') as f:
    for stop_word in f:
        stop_wordset.add(stop_word.strip('\n'))

print("Dictionary " + jieba_dictionary_name + " prepared.")


def cut(sentence, cut_all=False):
    return [word for word in jieba.cut(sentence, cut_all=cut_all)
            if len(word.strip()) != 0
            and word not in stop_wordset]


def extract_tag(sentence, withWeight=False):
    return [word for word in jieba.analyse.extract_tags(sentence, withWeight=withWeight)
            if len(word.strip()) != 0
            and word not in stop_wordset]

