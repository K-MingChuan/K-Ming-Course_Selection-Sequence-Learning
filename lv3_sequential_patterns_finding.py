from algorithms.prefixspan import PrefixSpan
from data_preprocessing import *
import operator

data = load_lv3_data()

model = PrefixSpan.train(data, minSupport=10)
result = sorted([(fs.sequence, fs.freq) for fs in model.freqSequences().collect()], key=operator.itemgetter(1), reverse=True)
for sequence, frequency in result:
    if len(sequence) >= 3:
        print('{}, {}'.format(sequence, frequency))

id_to_courses = load_students_course_set(['04362481'])