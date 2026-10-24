import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
import symnmf

MAX_ITER = 300
EPSILON = 1e-4

def handle_error():
    print("An Error Has Occurred")
    sys.exit(1)


def read_data_points(file_name):
    try:
        data = []
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append([float(x) for x in line.split(",")])
        return data
    except Exception:
        handle_error()


def distance(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))


def closest_centroid(p, centroids):
    min_dist = float("inf")
    closest = 0
    for idx, c in enumerate(centroids):
        dist = distance(p, c)
        if dist < min_dist:
            min_dist = dist
            closest = idx
    return closest


def run_kmeans(points, k):
    d = len(points[0])
    centroids = [points[i].copy() for i in range(k)]

    for _ in range(MAX_ITER):
        old_centroids = [c.copy() for c in centroids]
        clusters = [[] for _ in range(k)]

        for p in points:
            i = closest_centroid(p, centroids)
            clusters[i].append(p)

        for i in range(k):
            if len(clusters[i]) != 0:
                centroids[i] = [
                    sum(c[j] for c in clusters[i]) / len(clusters[i])
                    for j in range(d)
                ]
            else:
                centroids[i] = old_centroids[i]

        max_dist = max(distance(centroids[i], old_centroids[i]) for i in range(k))
        if max_dist < EPSILON:
            break

    labels = [closest_centroid(p, centroids) for p in points]
    return labels


def init_h(w_matrix, n, k):
    np.random.seed(1234)
    w_np = np.array(w_matrix, dtype=float)
    m = np.mean(w_np)
    upper_bound = 2.0 * np.sqrt(m / k)
    return np.random.uniform(0.0, upper_bound, size=(n, k)).tolist()


def run_symnmf(points, k):
    n = len(points)
    w_mat = symnmf.norm(points)
    h_init = init_h(w_mat, n, k)
    h_final = symnmf.symnmf(h_init, w_mat)
    # Cluster assignment: column index of maximum value per row
    labels = [int(np.argmax(row)) for row in h_final]
    return labels

def handle_cluster_error():
    print("Incorrect number of clusters!")
    sys.exit(1)


def main():
    if len(sys.argv) != 3:
        handle_error()

    try:
        k_val = float(sys.argv[1])
        if not k_val.is_integer():
            handle_cluster_error()
        k = int(k_val)
    except Exception:
        handle_cluster_error()

    file_name = sys.argv[2]
    points = read_data_points(file_name)
    n = len(points)

    if k <= 1 or k >= n:
        handle_cluster_error()
    labels_symnmf = run_symnmf(points, k)
    labels_kmeans = run_kmeans(points, k)

    # Handle edge case where clustering yields < 2 unique clusters
    try:
        if len(set(labels_symnmf)) < 2 or len(set(labels_kmeans)) < 2:
            handle_error()

        score_symnmf = silhouette_score(points, labels_symnmf)
        score_kmeans = silhouette_score(points, labels_kmeans)
        score_ari = adjusted_rand_score(labels_symnmf, labels_kmeans)

        print(f"nmf: {score_symnmf:.4f}")
        print(f"kmeans: {score_kmeans:.4f}")
        print(f"ari: {score_ari:.4f}")

    except Exception:
        handle_error()

if __name__ == "__main__":
    main()