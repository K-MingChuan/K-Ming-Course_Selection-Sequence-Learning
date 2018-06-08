import json

import numpy

if __name__ == '__main__':
    a = numpy.random.randint(0, 100, size=10)
    print(a)
    print(sorted(enumerate(a), key=lambda t: t[1]))
    b = sorted(enumerate(a), key=lambda t: t[1])
    a[[index for index, val in b[0:3]]] = 100
    print(a)