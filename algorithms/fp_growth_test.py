
if __name__ == '__main__':
    data = [['M', 'O', 'N', 'K', 'E', 'Y'],
            ['D', 'O', 'N', 'K', 'E', 'Y'],
            ['M', 'A', 'K', 'E'],
            ['M', 'U', 'C', 'K', 'Y'],
            ['C', 'O', 'O', 'K', 'I', 'E']]

    from fp_growth import find_frequent_itemsets

    for itemset in [l for l in find_frequent_itemsets(data, 3) if len(l) > 1]:
        print(itemset)
