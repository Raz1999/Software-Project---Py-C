import sys
import math
import numpy as np
import pandas as pd
import mykmeanssp

DEFAULT_ERROR = "An Error Has Occurred"
DEFAULT_K = 3
MAX_ITER = 400


# helper functions

def parse_arguments() -> tuple[int, int, float, str, str]:
    # arg validation
    if len(sys.argv) == 5:
        k = DEFAULT_K

        try:
            itr = int(sys.argv[1])
        except ValueError:
            print("Incorrect maximum iteration!")
            sys.exit(1)

        try:
            eps = float(sys.argv[2])
        except ValueError:
            print("Incorrect epsilon!")
            sys.exit(1)

        file1 = sys.argv[3]
        file2 = sys.argv[4]

    elif len(sys.argv) == 6:
        try:
            k = int(sys.argv[1])
        except ValueError:
            print("Incorrect number of clusters!")
            sys.exit(1)

        try:
            itr = int(sys.argv[2])
        except ValueError:
            print("Incorrect maximum iteration!")
            sys.exit(1)

        try:
            eps = float(sys.argv[3])
        except ValueError:
            print("Incorrect epsilon!")
            sys.exit(1)

        file1 = sys.argv[4]
        file2 = sys.argv[5]

    else: # wrong number of args
        print(DEFAULT_ERROR)
        sys.exit(1)

    # arg validation
    if k <= 1:
        print("Incorrect number of clusters!")
        sys.exit(1)
    if not (1 < itr < MAX_ITER):
        print("Incorrect maximum iteration!")
        sys.exit(1)
    if eps < 0 or not math.isfinite(eps):
        print("Incorrect epsilon!")
        sys.exit(1)

    return (k, itr, eps, file1, file2)

# merge file1 and file2 using pandas, then convert to python object
def load_data(file1:str, file2:str) -> tuple[list[int], list[list[float]]]:
    try:
        df1 = pd.read_csv(file1, header=None)
        df2 = pd.read_csv(file2, header=None)
        if df1.empty or df2.empty:
            raise ValueError

        # Data Validation of data in merged DF object:

        # check that all entries are numbers
        df1 = df1.apply(pd.to_numeric, errors="raise")
        df2 = df2.apply(pd.to_numeric, errors="raise")

        # flag missing cells or trailing commas.
        if df1.isnull().values.any() or df2.isnull().values.any():
            raise ValueError

        # reject infinity and NaN values
        for row in df1.values.tolist():
            for value in row:
                if not math.isfinite(value):
                    raise ValueError
        for row in df2.values.tolist():
            for value in row:
                if not math.isfinite(value):
                    raise ValueError

        joined = pd.merge(df1, df2, how="inner", on=0)
        if joined.empty:
            raise ValueError

        joined = joined.sort_values(by=0) #sort by key
        keys = joined[0].tolist()
        points = joined.drop(columns=0).values.tolist() #remove key column before converting to list

        # Data validation of python list
        if len(points) == 0 or len(points[0]) == 0:
            raise ValueError
        dimension = len(points[0])
        for point in points:
            if len(point) != dimension:
                raise ValueError

        return (keys, points)

    except Exception:
        print(DEFAULT_ERROR)
        sys.exit(1)


def initialize_centroids(points: list[list[float]], k: int) -> tuple[list[int], list[list[float]]]:
    np.random.seed(1234)
    n = len(points)
    points_array = np.array(points, dtype=float) # convert points into numpy array
    centroid_indices = []
    centroids = []

    # choose first centroid at random
    first = int(np.random.choice(n))
    centroid_indices.append(first)
    centroids.append(points_array[first])

    # choose remaining centroids
    while len(centroids) < k:
        distances = np.zeros(n) # makes array of n zeros, used to hold distances of steps 2-3

        # find distance of every point tp nearest centroid
        for i in range(n):
            min_dist = -1.0
            for centroid in centroids:
                dist = np.sqrt( np.sum((points_array[i] - centroid) ** 2) ) #euclidean distance
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
            distances[i] = min_dist

        probabilities = distances / np.sum(distances) # create probability array based on distances

        # choose the next centroid, previously chosen points have P=0 of being chosen
        next = int(np.random.choice(n, p=probabilities))
        centroid_indices.append(next)
        centroids.append(points_array[next])

    # convert the numpy centroids arrays into Python lists
    initial_centroids = [centroid.tolist() for centroid in centroids]

    return (centroid_indices, initial_centroids)


# print formatting for indices (line 1 of output)
def print_indices(indices: list[int]):
    formatted = str(indices[0])

    for index in indices[1:]:
        formatted += f",{index}"

    print(formatted)

# print formatting for centroids (line 2+ of output)
def print_centroids(centroids):
    for centroid in centroids:
        formatted = f"{centroid[0]:.4f}"

        for value in centroid[1:]:
            formatted += f",{value:.4f}"

        print(formatted)


def main():
    k, itr, eps, file1, file2 = parse_arguments()
    keys, points = load_data(file1, file2)

    # check k upper-bound
    n = len(points)
    if k >= n:
        print("Incorrect number of clusters!")
        sys.exit(1)

    # choose initial centroids and save their indices for printing
    indices, initial_centroids = initialize_centroids(points, k)

    # call C extension to run kmeans algorithm on points to update centroids
    try:
        centroids = mykmeanssp.fit(initial_centroids, points, itr, eps)
    except Exception:
        print(DEFAULT_ERROR)
        sys.exit(1)

    initial_keys = [int(keys[i]) for i in indices] # recover keys for chosen initial centroids
    print_indices(initial_keys)
    print_centroids(centroids)

if __name__ == "__main__":
    main()