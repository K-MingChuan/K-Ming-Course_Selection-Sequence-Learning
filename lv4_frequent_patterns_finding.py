

from data_preprocessing import load_lv4_data, load_lv4_cluster_idxs_records
from algorithms.fp_growth import find_frequent_itemsets

if __name__ == '__main__':
    """
    input: 
    output:
    """

    taken_course_names_of_students, clusters = load_lv4_data()
    cluster_idxs_records = \
        load_lv4_cluster_idxs_records(taken_course_names_of_students, clusters)
    # Applying FP Growth algorithm to the records
    print('Start doing fp-growth...')
    # If the support is too low (<1000), then it will take about 10 mins.
    support = 15000
    print('Minsup.: {}'.format(support))
    cluster_patterns = \
        list(find_frequent_itemsets(cluster_idxs_records,
                                    include_support=True,
                                    minimum_support=support))
    # We're interested in patterns in descending support, to recommend courses in
    #   the mainstream course-cluster.
    cluster_patterns.sort(reverse=True, key=lambda item: item[1])
    print(cluster_patterns)


