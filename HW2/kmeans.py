import sys
import math


DEFAULT_ERROR = "An Error Has Occured"
DEFAULT_ITER = 400
MAX_ITER = 800
EPSILON = 0.001


#helper funtions
def distance(p1:list[float],p2:list[float]) -> float:
    d = 0
    for i in range(len(p1)):
        d += (p1[i]-p2[i])**2
    d = math.sqrt(d)
    return d

def closest_centroid(p:list[float],centroids:list[list[float]]) -> int:
    min_dist = -1
    closest = 0
    for i in range(len(centroids)):
        if (min_dist == -1 or distance(p,centroids[i]) < min_dist):
            min_dist = distance(p,centroids[i])
            closest = i
    return closest


def main():
    #arg validation
    if len(sys.argv) not in (2,3):
        print(DEFAULT_ERROR)
        sys.exit(1)
    try:
        k = int(sys.argv[1]) #centroid count
        if len(sys.argv) == 3:
            itr = int(sys.argv[2]) #iterations
        else:
            itr = DEFAULT_ITER
    except:
        print(DEFAULT_ERROR)
        sys.exit(1)
    if not (1 < itr < MAX_ITER):
        print("Incorrect maximum iteration!")
        sys.exit(1)

    #stdin data validation/entry
    points = []
    try:
        for line in sys.stdin:
            p = [float(x) for x in line.split(",")]
            for x in p:
                if not math.isfinite(x):
                    raise ValueError
            points.append(p)
    except Exception:
        print(DEFAULT_ERROR)
        sys.exit(1)
    n = len(points) #set size
    if n == 0:
        print(DEFAULT_ERROR)
        sys.exit(1)
    if not (1 < k < n):
        print("Incorrect number of clusters!")
        sys.exit(1)
    d = len(points[0]) #dimension
    for p in points:
        if len(p) != d:
            print(DEFAULT_ERROR)
            sys.exit(1)

    #kmeans algorithm
    centroids = [points[i] for i in range(k)]
    #assign clusters
    for loop in range(itr):
        old_centroids = centroids.copy()
        clusters = [[] for i in range(k)]
        for p in points:
            i = closest_centroid(p,centroids)
            clusters[i].append(p)
    #update centroids
        centroids = [[] for i in range(k)]
        for i in range(k):
            new_centroid = [0.0 for i in range(d)]
            for c in clusters[i]:
                for j in range(d):
                    new_centroid[j] += c[j]
            for j in range(d):
                new_centroid[j] = new_centroid[j]/len(clusters[i]) if len(clusters[i]) != 0 else old_centroids[i][j]
            centroids[i] = new_centroid
    #check change in centroids
        max_dist = 0
        for i in range(k):
            if distance(centroids[i],old_centroids[i]) > max_dist:
                max_dist = distance(centroids[i],old_centroids[i])
        if max_dist < EPSILON:
            break

    #print centroids
    for c in centroids:
        formatted = f"{c[0]:.4f}"
        for val in c[1:]:
            formatted += (f",{val:.4f}")
        print(formatted)

if __name__ == "__main__":
    main()