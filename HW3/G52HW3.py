import sys
import threading
import random
from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.streaming import StreamingContext

#a big prime number for hash functions
PRIME = 8191 

def generate_hash_functions(D, W):
    return [(random.randint(1, PRIME-1), random.randint(0, PRIME-1)) for _ in range(D)]

def hash_fn(a, b, x, W):
    return ((a * x + b) % PRIME) % W

def sign_hash_fn(a, b, x):
    return 1 if ((a * x + b) % PRIME) % 2 == 0 else -1

def process_batch(time, rdd):
    global streamLength, histogram, CM, CS, D, W, CM_hashes, CS_hashes_h, CS_hashes_g

    items = rdd.map(lambda x: int(x)).collect()
    if not items:
        return

    batch_size = len(items)
    if streamLength[0] >= THRESHOLD:
        return

    streamLength[0] += batch_size

    #update exact counts
    for item in items:
        histogram[item] = histogram.get(item, 0) + 1

        for i in range(D):
            col = hash_fn(CM_hashes[i][0], CM_hashes[i][1], item, W)
            CM[i][col] += 1

            col_cs = hash_fn(CS_hashes_h[i][0], CS_hashes_h[i][1], item, W)
            sign = sign_hash_fn(CS_hashes_g[i][0], CS_hashes_g[i][1], item)
            CS[i][col_cs] += sign

    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()

def estimate_cm(item):
    return min(CM[i][hash_fn(CM_hashes[i][0], CM_hashes[i][1], item, W)] for i in range(D))

def estimate_cs(item):
    estimates = [CS[i][hash_fn(CS_hashes_h[i][0], CS_hashes_h[i][1], item, W)] *
                 sign_hash_fn(CS_hashes_g[i][0], CS_hashes_g[i][1], item) for i in range(D)]
    return sorted(estimates)[len(estimates)//2]  #median

if __name__ == '__main__':

    portExp = int(sys.argv[1])
    THRESHOLD = int(sys.argv[2])
    D = int(sys.argv[3])
    W = int(sys.argv[4])
    K = int(sys.argv[5])

    conf = SparkConf().setMaster("local[*]").setAppName("G52HW3")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.05)
    ssc.sparkContext.setLogLevel("ERROR")

    streamLength = [0]
    histogram = {}
    CM = [[0] * W for _ in range(D)]
    CS = [[0] * W for _ in range(D)]
    CM_hashes = generate_hash_functions(D, W)
    CS_hashes_h = generate_hash_functions(D, W)
    CS_hashes_g = generate_hash_functions(D, W)

    #stream
    stopping_condition = threading.Event()
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    stream.foreachRDD(lambda time, rdd: process_batch(time, rdd))

    #run stream
    ssc.start()
    stopping_condition.wait()
    ssc.stop(False, False)

    print(f"Port = {portExp} T = {THRESHOLD} D = {D} W = {W} K = {K}")
    print(f"Number of processed items = {streamLength[0]}")
    print(f"Number of distinct items  = {len(histogram)}")

    #top-K true heavy hitters
    true_freqs = sorted(histogram.items(), key=lambda x: -x[1])
    phi = true_freqs[K-1][1]
    top_k_items = [item for item, freq in true_freqs if freq >= phi]
    print(f"Number of Top-K Heavy Hitters = {len(top_k_items)}")

    #compute errors
    def rel_error(true_f, est_f):
        return abs(est_f - true_f) / true_f if true_f > 0 else 0

    cm_errors = [rel_error(histogram[u], estimate_cm(u)) for u in top_k_items]
    cs_errors = [rel_error(histogram[u], estimate_cs(u)) for u in top_k_items]

    avg_cm_error = sum(cm_errors) / len(cm_errors)
    avg_cs_error = sum(cs_errors) / len(cs_errors)

    print(f"Avg Relative Error for Top-K Heavy Hitters with CM = {avg_cm_error}")
    print(f"Avg Relative Error for Top-K Heavy Hitters with CS = {avg_cs_error}")

    if K <= 10:
        print("Top-K Heavy Hitters:")
        for u in sorted(top_k_items):
            true_f = histogram[u]
            est_f_cm = estimate_cm(u)
            print(f"Item {u} True Frequency = {true_f} Estimated Frequency with CM = {est_f_cm}")
