import sys
import numpy as np
import symnmf

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


def print_matrix(matrix):
    for row in matrix:
        print(",".join(f"{val:.4f}" for val in row))


def init_h(w_matrix, n, k):
    np.random.seed(1234)
    w_np = np.array(w_matrix, dtype=float)
    m = np.mean(w_np)
    upper_bound = 2.0 * np.sqrt(m / k)
    h = np.random.uniform(0.0, upper_bound, size=(n, k))
    return h.tolist()

def handle_cluster_error():
    print("Incorrect number of clusters!")
    sys.exit(1)


def main():
    if len(sys.argv) != 4:
        handle_error()

    try:
        k_val = float(sys.argv[1])
        if not k_val.is_integer():
            handle_cluster_error()
        k = int(k_val)
    except Exception:
        handle_cluster_error()

    goal = sys.argv[2]
    file_name = sys.argv[3]

    if goal not in ["symnmf", "sym", "ddg", "norm"]:
        handle_error()

    points = read_data_points(file_name)
    n = len(points)

    if k <= 1 or k >= n:
        handle_cluster_error()

    result = None

    if goal == "sym":
        result = symnmf.sym(points)
    elif goal == "ddg":
        result = symnmf.ddg(points)
    elif goal == "norm":
        result = symnmf.norm(points)
    elif goal == "symnmf":
        w_mat = symnmf.norm(points)
        h_init = init_h(w_mat, n, k)
        result = symnmf.symnmf(h_init, w_mat)
    else:
        handle_error()

    if result is None:
        handle_error()

    print_matrix(result)


if __name__ == "__main__":
    main()

    