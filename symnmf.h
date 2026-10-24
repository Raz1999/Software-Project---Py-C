#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Matrix allocation and deallocation */
double **allocate_matrix(int rows, int cols);
void free_matrix(double **matrix, int rows);
void print_matrix(double **matrix, int rows, int cols);

/* Main mathematical functions */
double **compute_similarity_matrix(double **X, int n, int d);
double **compute_degree_matrix(double **A, int n);
double **compute_normalized_matrix(double **A, double **D, int n);
void update_h_matrix(double **H, double **W, int n, int k);


/*C API */
double **sym(double **X, int n, int d);
double **ddg(double **X, int n, int d);
double **norm(double **X, int n, int d);

#endif 