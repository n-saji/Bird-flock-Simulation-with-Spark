from itertools import combinations
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd

def ComputeDistanceWithSpark(pair):
    spark = SparkSession.builder.appName("Edit-Distance").getOrCreate()
    dataFrame = spark.createDataFrame(pair,["str1", "str2"])
    @pandas_udf("int")
    def editUDFDistance(str1: pd.Series, str2: pd.Series) -> pd.Series:
        return pd.Series([edit_distance((s1, s2)) for s1, s2 in zip(str1, str2)])
    _ = dataFrame.withColumn("distances", editUDFDistance(dataFrame["str1"], dataFrame["str2"]))
    spark.stop()
    return

def edit_distance(pair):
    str1, str2 = pair
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j 
            elif j == 0:
                dp[i][j] = i 
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], 
                                   dp[i - 1][j],  
                                   dp[i - 1][j - 1]) 

    return dp[m][n]

def compute_edit_distance_multiprocess(pair, num_workers):
    with multiprocessing.Pool(num_workers) as pool:
        distances = list(tqdm(pool.imap(edit_distance, pair), total=len(pair), ncols=100))
    return distances

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--csv_dir', type=str, default='simple-wiki-unique-has-end-punct-sentences.csv', help="Directory of csv file")
    parser.add_argument('--num_sentences', type=int, default=300, help="Number of sentences")
    args = parser.parse_args()
    num_workers = multiprocessing.cpu_count()
    print(f'number of available cpu cores: {num_workers}')
    text_data = pd.read_csv(args.csv_dir)['sentence']
    text_data = text_data[:args.num_sentences]
    pair_data = list(combinations(text_data, 2))
    # Spark
    start_time = time.time()
    _ = ComputeDistanceWithSpark(pair_data)
    end_time = time.time()
    time1 = end_time - start_time
    print(f"Time taken (Spark): {end_time - start_time:.2f} seconds")
    # Multi-process
    start_time = time.time()
    edit_distances = compute_edit_distance_multiprocess(pair_data, num_workers)
    end_time = time.time()
    time2 = end_time - start_time
    print(f"Time taken (multi-process): {end_time - start_time:.3f} seconds")
    # Vanilla for loop
    start_time = time.time()
    distances = []
    for pair in tqdm(pair_data, ncols=100):
        distances.append(edit_distance(pair))
    end_time = time.time()
    time3 = end_time - start_time
    print(f"Time taken (for-loop): {end_time - start_time:.3f} seconds")
    print(f"Time cost (Spark, multi-process, for-loop): [{time1:.3f}, {time2:.3f}, {time3:.3f}]")