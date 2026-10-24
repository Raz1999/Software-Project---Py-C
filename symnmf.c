#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

#define MAX_ITER 300
#define EPSILON 1e-4
#define BETA 0.5

/* Error handling function */
void handle_error(void) {
    printf("An Error Has Occurred\n");
    exit(1);
}

/* Function to allocate a 2D matrix */
double **allocate_matrix(int rows, int cols) {
    int i, j;
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (mat == NULL) {
        handle_error();
    }
    for (i = 0; i < rows; i++) {
        mat[i] = (double *)calloc(cols, sizeof(double));
        if (mat[i] == NULL) {
            for (j = 0; j < i; j++) {
                free(mat[j]);
            }
            free(mat);
            handle_error();
        }
    }
    return mat;
}

/* Function to free a 2D matrix */
void free_matrix(double **matrix, int rows) {
    int i;
    if (matrix == NULL) {
        return;
    }
    for (i = 0; i < rows; i++) {
        if (matrix[i] != NULL) {
            free(matrix[i]);
        }
    }
    free(matrix);
}

/* Function to print a 2D matrix */
void print_matrix(double **matrix, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

/* Function to compute the squared Euclidean distance between two vectors */
double squared_euclidean_distance(double *p, double *q, int d) {
    int i;
    double dist = 0.0, diff;
    for (i = 0; i < d; i++) {
        diff = p[i] - q[i];
        dist += diff * diff;
    }
    return dist;
}

/* Function to multiply two matrices A (n x m) and B (m x p) and store the result in C (n x p) */
void multiply_matrices(double **A, double **B, double **C, int n, int m, int p) {
    int i, j, l;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (l = 0; l < m; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

/* Function to compute the Frobenius norm of the difference between two matrices H_new and H_old */
double frobenius_norm_diff_sq(double **H_new, double **H_old, int n, int k) {
    int i, j;
    double sum = 0.0, diff;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            diff = H_new[i][j] - H_old[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

/* Function to compute the similarity matrix A from the input data matrix X */
double **compute_similarity_matrix(double **X, int n, int d) {
    int i, j;
    double **A = allocate_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                A[i][j] = 0.0;
            } else {
                A[i][j] = exp(-squared_euclidean_distance(X[i], X[j], d) / 2.0);
            }
        }
    }
    return A;
}

/* Function to compute the degree matrix D from the similarity matrix A */
double **compute_degree_matrix(double **A, int n) {
    int i, j;
    double **D = allocate_matrix(n, n);
    for (i = 0; i < n; i++) {
        double sum = 0.0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        D[i][i] = sum;
    }
    return D;
}

/* Function to compute the normalized matrix W from the similarity matrix A and degree matrix D */
double **compute_normalized_matrix(double **A, double **D, int n) {
    int i, j;
    double **W = allocate_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (D[i][i] > 0.0 && D[j][j] > 0.0) {
                W[i][j] = A[i][j] / sqrt(D[i][i] * D[j][j]);
            } else {
                W[i][j] = 0.0;
            }
        }
    }
    return W;
}

/* C API functions */

/* Function to compute the similarity matrix A from the input data matrix X */
double **sym(double **X, int n, int d) {
    return compute_similarity_matrix(X, n, d);
}

/* Function to compute the degree matrix D from the input data matrix X */
double **ddg(double **X, int n, int d) {
    double **A = compute_similarity_matrix(X, n, d);
    double **D = compute_degree_matrix(A, n);
    free_matrix(A, n);
    return D;
}

/* Function to compute the normalized matrix W from the input data matrix X */
double **norm(double **X, int n, int d) {
    double **A = compute_similarity_matrix(X, n, d);
    double **D = compute_degree_matrix(A, n);
    double **W = compute_normalized_matrix(A, D, n);
    free_matrix(A, n);
    free_matrix(D, n);
    return W;
}

/* Function to transpose a matrix H (n x k) and store the result in H_T (k x n) */
void transpose(double **H, double **H_T, int n, int k) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            H_T[j][i] = H[i][j];
        }
    }
}

/* Helper to perform single entry update rule */
static void compute_next_h(double **H, double **H_next, double **WH, double **HHtH, int n, int k) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            double denom = HHtH[i][j];
            double num = WH[i][j];
            double factor = (denom != 0.0) ? (num / denom) : 0.0;
            H_next[i][j] = H[i][j] * (1.0 - BETA + BETA * factor);
        }
    }
}

/* Performs updates in-place */
void update_h_matrix(double **H, double **W, int n, int k) {
    int iter, i, j;
    double **H_next = allocate_matrix(n, k);
    double **WH = allocate_matrix(n, k), **H_T = allocate_matrix(k, n);
    double **HtH = allocate_matrix(k, k), **HHtH = allocate_matrix(n, k);

    for (iter = 0; iter < MAX_ITER; iter++) {
        multiply_matrices(W, H, WH, n, n, k);
        transpose(H, H_T, n, k);
        multiply_matrices(H_T, H, HtH, k, n, k);
        multiply_matrices(H, HtH, HHtH, n, k, k);

        compute_next_h(H, H_next, WH, HHtH, n, k);

        if (frobenius_norm_diff_sq(H_next, H, n, k) < EPSILON) {
            for (i = 0; i < n; i++) {
                for (j = 0; j < k; j++) H[i][j] = H_next[i][j];
            }
            break;
        }
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) H[i][j] = H_next[i][j];
        }
    }

    free_matrix(H_next, n);
    free_matrix(WH, n);
    free_matrix(H_T, k);
    free_matrix(HtH, k);
    free_matrix(HHtH, n);
}

/* Function that counts the number of rows and columns in a file */
void get_dimensions(const char *filename, int *n, int *d) {
    FILE *file = fopen(filename, "r");
    int ch, prev = '\n';
    int rows = 0, cols = 1, first_row_done = 0;

    if (file == NULL) {
        handle_error();
    }

    while ((ch = fgetc(file)) != EOF) {
        if (!first_row_done && ch == ',') {
            cols++;
        }
        if (ch == '\n') {
            if (prev != '\n' && prev != '\r') {
                rows++;
                first_row_done = 1;
            }
        }
        prev = ch;
    }
    if (prev != '\n' && prev != '\r' && prev != EOF) {
        rows++;
    }

    fclose(file);
    *n = rows;
    *d = cols;
}

/* Reads metrix X from file given known n and d */
double **read_data_points(const char *filename, int n, int d) {
    FILE *file = fopen(filename, "r");
    int i, j;
    double **X;

    if (file == NULL) {
        handle_error();
    }

    X = allocate_matrix(n, d);
    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            if (fscanf(file, "%lf", &X[i][j]) != 1) {
                free_matrix(X, i + 1);
                fclose(file);
                handle_error();
            }
            fgetc(file); /* consumes ',' or '\n' */
        }
        /* skip trailing carriage returns or empty newlines */
        while (1) {
            int c = fgetc(file);
            if (c == EOF || (c != '\n' && c != '\r' && c != ' ')) {
                if (c != EOF) ungetc(c, file);
                break;
            }
        }
    }
    fclose(file);
    return X;
}

#ifndef FROM_MODULE
int main(int argc, char **argv) {
    char *goal;
    char *filename;
    int n = 0, d = 0;
    double **X = NULL, **result = NULL;

    if (argc != 3) {
        handle_error();
    }

    goal = argv[1];
    filename = argv[2];

    get_dimensions(filename, &n, &d);
    X = read_data_points(filename, n, d);

    if (strcmp(goal, "sym") == 0) {
        result = sym(X, n, d);
    } else if (strcmp(goal, "ddg") == 0) {
        result = ddg(X, n, d);
    } else if (strcmp(goal, "norm") == 0) {
        result = norm(X, n, d);
    } else {
        free_matrix(X, n);
        handle_error();
    }

    print_matrix(result, n, n);

    free_matrix(X, n);
    free_matrix(result, n);
    return 0;
}
#endif