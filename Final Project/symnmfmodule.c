#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Converts a Python list to a 2D matrix, to double** */
static double **py_list_to_matrix(PyObject *list_obj, int *rows, int *cols) {
    int i, j;
    double **mat;
    *rows = (int)PyList_Size(list_obj);
    if (*rows == 0) return NULL;
    *cols = (int)PyList_Size(PyList_GetItem(list_obj, 0));

    mat = allocate_matrix(*rows, *cols);
    for (i = 0; i < *rows; i++) {
        PyObject *row_obj = PyList_GetItem(list_obj, i);
        for (j = 0; j < *cols; j++) {
            PyObject *item = PyList_GetItem(row_obj, j);
            mat[i][j] = PyFloat_AsDouble(item);
        }
    }
    return mat;
}

/* Converts a 2D matrix to a Python list and free C matrix */
static PyObject *matrix_to_py_list(double **mat, int rows, int cols) {
    int i, j;
    PyObject *outer_list = PyList_New(rows);
    if (!outer_list) return NULL;

    for (i = 0; i < rows; i++) {
        PyObject *row_list = PyList_New(cols);
        for (j = 0; j < cols; j++) {
            PyList_SetItem(row_list, j, PyFloat_FromDouble(mat[i][j]));
        }
        PyList_SetItem(outer_list, i, row_list);
    }
    free_matrix(mat, rows);
    return outer_list;
}


static PyObject *py_sym(PyObject *self, PyObject *args) {
    PyObject *x_obj;
    int n, d;
    double **X, **A;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) return NULL;
    X = py_list_to_matrix(x_obj, &n, &d);
    A = sym(X, n, d);
    free_matrix(X, n);
    return matrix_to_py_list(A, n, n);
}

static PyObject *py_ddg(PyObject *self, PyObject *args) {
    PyObject *x_obj;
    int n, d;
    double **X, **D;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) return NULL;
    X = py_list_to_matrix(x_obj, &n, &d);
    D = ddg(X, n, d);
    free_matrix(X, n);
    return matrix_to_py_list(D, n, n);
}

static PyObject *py_norm(PyObject *self, PyObject *args) {
    PyObject *x_obj;
    int n, d;
    double **X, **W;
    (void)self;

    if (!PyArg_ParseTuple(args, "O", &x_obj)) return NULL;
    X = py_list_to_matrix(x_obj, &n, &d);
    W = norm(X, n, d);
    free_matrix(X, n);
    return matrix_to_py_list(W, n, n);
}

static PyObject *py_symnmf(PyObject *self, PyObject *args) {
    PyObject *h_obj, *w_obj;
    int n_h, k, n_w, n_w2;
    double **H, **W;
    (void)self;

    if (!PyArg_ParseTuple(args, "OO", &h_obj, &w_obj)) return NULL;
    H = py_list_to_matrix(h_obj, &n_h, &k);
    W = py_list_to_matrix(w_obj, &n_w, &n_w2);

    update_h_matrix(H, W, n_h, k);
    free_matrix(W, n_w);
    return matrix_to_py_list(H, n_h, k);
}

static PyMethodDef symnmf_methods[] = {
    {"sym", py_sym, METH_VARARGS, "Calculate similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Calculate diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Calculate normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Calculate full SymNMF H matrix"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "Python C API for SymNMF",
    -1,
    symnmf_methods
};

PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}

