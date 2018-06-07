import json

if __name__ == '__main__':
    a = [1, 2, 3, 4, 5]
    with open('test.json', 'w+', encoding='utf-8') as fw:
        json.dump(a, fw)