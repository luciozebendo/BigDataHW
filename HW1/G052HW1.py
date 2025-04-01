import sys
from pyspark.sql import SparkSession
import numpy as np

def parse_line(line):
    #Parses a line into (point, group).
    parts = line.strip().split(',')
    return (tuple(map(float, parts[:-1])), parts[-1])

def compute_centroids(data_rdd, k, max_iterations):
    #Runs k-means to find K centroids.
    centroids = data_rdd.map(lambda x: x[0]).takeSample(False, k)

    for _ in range(max_iterations):
        clusters = data_rdd.map(lambda x: (
            min(range(k), key=lambda i: np.linalg.norm(np.array(x[0]) - np.array(centroids[i]))), x
        ))
        new_centroids = clusters.groupByKey().mapValues(
            lambda points: tuple(np.mean([p[0] for p in points], axis=0))
        ).collectAsMap()

        for i in new_centroids:
            centroids[i] = new_centroids[i]

    return centroids

def MRComputeStandardObjective(data_rdd, centroids):
    #Computes standard k-means objective Delta(U,C).
    def closest_centroid(point):
        return min(range(len(centroids)), key=lambda i: np.linalg.norm(np.array(point) - np.array(centroids[i])))

    return data_rdd.map(lambda x: np.linalg.norm(np.array(x[0]) - np.array(centroids[closest_centroid(x[0])])) ** 2).mean()

def MRComputeFairObjective(data_rdd, centroids):
    #Computes fair objective Phi(A,B,C).
    grouped_data = data_rdd.groupBy(lambda x: x[1]).mapValues(list).collectAsMap()
    A_points = sc.parallelize(grouped_data.get('A', []))
    B_points = sc.parallelize(grouped_data.get('B', []))

    def compute_fair_objective(points_rdd):
        return points_rdd.map(lambda x: np.linalg.norm(np.array(x[0]) - np.array(centroids[min(range(len(centroids)), key=lambda i: np.linalg.norm(np.array(x[0]) - np.array(centroids[i])))])) ** 2).mean()

    return max(compute_fair_objective(A_points), compute_fair_objective(B_points))

def MRPrintStatistics(data_rdd, centroids):
    #Prints point distribution across centroids.
    def closest_centroid(point):
        return min(range(len(centroids)), key=lambda i: np.linalg.norm(np.array(point) - np.array(centroids[i])))

    cluster_stats = data_rdd.map(lambda x: (closest_centroid(x[0]), x[1])).groupByKey().mapValues(lambda points: (sum(1 for p in points if p == 'A'), sum(1 for p in points if p == 'B'))).collect()

    for cluster_id, (NA, NB) in cluster_stats:
        print(f"Centroid {cluster_id}: A={NA}, B={NB}")

spark = SparkSession.builder.appName("KMeansClustering").getOrCreate()
sc = spark.sparkContext
spark.stop()
