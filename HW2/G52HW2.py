import numpy as np
from pyspark.mllib.clustering import KMeans
from pyspark import SparkContext, SparkConf
# from G52HW1 import MRComputeFairObjective
import sys
import time
import random

def computeVectorX(fixed_a, fixed_b, alpha, beta, ell, k):
    gamma = 0.5
    x_dist = [0.0] * k
    power = 0.5
    t_max = 10

    for _ in range(t_max):
        f_a = fixed_a
        f_b = fixed_b
        power /= 2

        for i in range(k):
            temp = (1 - gamma) * beta[i] * ell[i] / (gamma * alpha[i] + (1 - gamma) * beta[i])
            x_dist[i] = temp
            f_a += alpha[i] * temp * temp
            temp = ell[i] - temp
            f_b += beta[i] * temp * temp

        if f_a == f_b:
            break

        gamma = gamma + power if f_a > f_b else gamma - power

    return x_dist

# PT1
def MRFairLloyd(inputPoints, K, M):
    total_A = inputPoints.filter(lambda x: x[1] == 'A').count()
    total_B = inputPoints.filter(lambda x: x[1] == 'B').count()

    model = KMeans.train(inputPoints.map(lambda x: x[0]), K, maxIterations=0)
    centroids = model.clusterCenters

    for iteration in range(M):
        centroid_bcast = inputPoints.context.broadcast(centroids)

        # Assign each point to nearest centroid
        clusters = inputPoints.map(lambda point: (
            np.argmin([np.linalg.norm(np.array(point[0]) - np.array(c)) for c in centroid_bcast.value]),
            point
        ))

        clustered_points = clusters.groupByKey().mapValues(list).collectAsMap()
        new_centroids = []

        for i in range(K):
            cluster_points = clustered_points.get(i, [])
            if not cluster_points:
                new_centroids.append(centroids[i])
                continue

            A_points = [np.array(p[0]) for p in cluster_points if p[1] == 'A']
            B_points = [np.array(p[0]) for p in cluster_points if p[1] == 'B']
            count_A = len(A_points)
            count_B = len(B_points)

            # Skip fairness if either group has too few samples (to avoid instability)
            if count_A < 5 or count_B < 5:
                mean = np.mean([np.array(p[0]) for p in cluster_points], axis=0)
                new_centroids.append(tuple(mean))
                continue

            mu_A = np.mean(A_points, axis=0)
            mu_B = np.mean(B_points, axis=0)
            ell = np.linalg.norm(mu_A - mu_B)

            alpha = count_A / total_A if total_A > 0 else 0.0
            beta = count_B / total_B if total_B > 0 else 0.0

            alpha_list = [alpha]
            beta_list = [beta]
            ell_list = [ell]

            delta_A = sum(np.linalg.norm(p - mu_A) ** 2 for p in A_points)
            delta_B = sum(np.linalg.norm(p - mu_B) ** 2 for p in B_points)
            fixedA = delta_A / count_A if count_A > 0 else 0.0
            fixedB = delta_B / count_B if count_B > 0 else 0.0

            x = computeVectorX(fixedA, fixedB, alpha_list, beta_list, ell_list, 1)[0]

            if ell == 0:
                new_c = mu_A
            else:
                # Precise interpolation from the slides
                new_c = ((ell - x) * mu_A + x * mu_B) / ell

            new_centroids.append(tuple(new_c))

        centroid_bcast.unpersist()
        centroids = new_centroids

    return centroids

# PT2: to imported from HM1 but i tried to improve it.
def MRComputeFairObjective(data_rdd, centroids):
    def closest_centroid(point):
        return min(range(len(centroids)), key=lambda i: np.linalg.norm(np.array(point) - np.array(centroids[i])))

    def compute_group_objective(group_label):
        return data_rdd.filter(lambda x: x[1] == group_label) \
                       .map(lambda x: np.linalg.norm(np.array(x[0]) - np.array(centroids[closest_centroid(x[0])])) ** 2).mean()

    phi_A = compute_group_objective('A')
    phi_B = compute_group_objective('B')

    return max(phi_A, phi_B)

# PT3
if __name__ == "__main__":

    # Create a Spark context
    conf = SparkConf().setAppName("G52HW2.py")
    conf.set("spark.locality.wait", "0s")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    input_path = sys.argv[1]
    L = int(sys.argv[2])
    K = int(sys.argv[3])
    M = int(sys.argv[4])

    print(f"Input file = {sys.argv[1]}, L = {sys.argv[2]}, K = {sys.argv[3]}, M = {sys.argv[4]}")

    #Load and parse data
    data = sc.textFile(input_path).repartition(L)
    parsed_data = data.map(lambda line: (tuple(map(float, line.strip().split(',')[:-1])), line.strip().split(',')[-1]))

    #Get counts
    N = parsed_data.count()
    NA = parsed_data.filter(lambda x: x[1] == 'A').count()
    NB = parsed_data.filter(lambda x: x[1] == 'B').count()

    print(f"N = {N}, NA = {NA}, NB = {NB}")

    #Lloyd's algorithm
    start_time = time.time()
    model = KMeans.train(parsed_data.map(lambda x: x[0]), K, maxIterations=M)
    C_max = model.clusterCenters
    time_max = time.time() - start_time

    #Fair K-Means
    start_time = time.time()
    C_fair = MRFairLloyd(parsed_data, K, M)
    time_fair = time.time() - start_time

    #Calculate objectives
    start_time = time.time()
    phi_max = MRComputeFairObjective(parsed_data, C_max)
    time_phi_max = time.time() - start_time

    start_time = time.time()
    phi_fair = MRComputeFairObjective(parsed_data, C_fair)
    time_phi_fair = time.time() - start_time

    #Print results
    print(f"Fair Objective with Standard Centers = {phi_max:.4f}")
    print(f"Fair Objective with Fair Centers = {phi_fair:.4f}")

    print(f"Time to compute standard centers = {int(time_max*1000)} ms")
    print(f"Time to compute fair centers = {int(time_fair*1000)} ms")
    print(f"Time to compute objective with standard centers = {int(time_phi_max*1000)} ms")
    print(f"Time to compute objective with fair centers = {int(time_phi_fair*1000)} ms")

    sc.stop()
