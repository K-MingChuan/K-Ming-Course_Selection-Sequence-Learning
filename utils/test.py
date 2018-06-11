
if __name__ == '__main__':
    data = [['A', 2, 3],
            ['A', 2, 4],
            [5, 6],
            [5],
            [8],
            ['A', 2, 3, 7, 5, 6],
            ['A'],
            ['A'],
            [1, 2, 3, 4, 5],
            [7, 5, 6, 7, 6]]

    from fp_growth import find_frequent_itemsets

    for itemset in [l for l in find_frequent_itemsets(data, 3) if len(l) >= 1]:
        print(itemset)
