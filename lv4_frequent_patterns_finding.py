

from data_preprocessing import load_lv4_data, compute_lv4_frequent_patterns

if __name__ == '__main__':
    # If the support is too low (<1000), then it will take about 10 mins.
    cluster_patterns = compute_lv4_frequent_patterns(support=12000, rebuild=True)
    cluster_patterns = compute_lv4_frequent_patterns(support=10000)
    cluster_patterns = compute_lv4_frequent_patterns(support=5000)
    print(cluster_patterns)


