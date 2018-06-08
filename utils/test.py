
if __name__ == '__main__':
    data = [[1, 2, 3],
            [1, 2, 4],
            [5, 6],
            [5],
            [8],
            [1, 2, 3, 7, 5, 6],
            [1, 2, 3, 4, 5],
            [7, 5, 6, 7, 6]]

    from fp_growth import find_frequent_itemsets

    for itemset in find_frequent_itemsets(data, 3):
        print(itemset)
