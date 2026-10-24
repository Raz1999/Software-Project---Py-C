#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>



/* חישוב המרחק האוקלידי בין שתי נקודות*/
static double euclidean_distance(double *a, double *b, int dim) {
    double sum = 0.0;
    int i;
    for(i = 0; i < dim; i++){
        /* מכפילים איבר בעצמו במקום להשתמש בחזקה*/
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

/* רצה על הצנטרואידים ומחזיקה את האינדקס של הקרוב ביותר*/
static int nearest_centroid(double *point, double **centroids, int K, int dim) {
    int k;
    int best = 0; /* מגדירים תחילה שהראשון הוא הכי קרוב*/
    double dist;
    /*מחשבים את המרחק לראשון ומגדירים כטוב ביותר*/
    double best_dist = euclidean_distance(point, centroids[0], dim);
    /* עוברים על שאר הצנטרואידים מ1 עד K - 1*/
    for(k = 1; k < K; k++) {
        dist = euclidean_distance(point, centroids[k], dim);
        /* אם מצאנו אחד קרוב יותר, מעדכנים אותו ואת האינדקס שלו*/
        if(dist < best_dist) {
            best_dist = dist;
            best = k;
        }
    }
    return best; /* מחזירים את האינדקס של הצנטרואיד*/
}

/* פונקציה לשחרור זיכרון המטריצה*/
/* קודם משחררים את המצביעים בתוך המטריצה ורק אחר כך אותה*/
static void free_matrix(double **matrix, int rows) {
    int i;
    /* אם המטריצה לא הוקצתה, לא עושה כלום*/
    if (matrix == NULL) {
        return;
    }

    /* עוברים שורה שורה ומשחררים את הקואורדינטות*/
    for (i = 0; i < rows; i++){
        if (matrix[i] != NULL){
            free(matrix[i]);
        }
    }

    /* משחררים את המטריצה הראשית*/
    free(matrix);
}

/* The function returns: -1 for allocation failure, 0 when not converged, and 1 when converged. */
static int update_centroids(double **points, double **centroids, int N, int K,
                            int dim, double epsilon) {
    double **sums;
    int *counts;
    int i, j, k;
    int cluster;
    int converged = 1;
    double change, new_value;

    /* מקצים טבלת סכומים*/
    sums = (double **)malloc(K * sizeof(double *));
    if (sums == NULL){
        return -1;
    }

    /* מקצים מערך מונים מאופס*/
    counts = (int *)calloc(K, sizeof(int));
    if (counts == NULL){
        free(sums);
        return -1;
    }

    for (k = 0; k < K; k++){
        sums[k] = (double *)calloc(dim, sizeof(double));
        if (sums[k] == NULL){
            free_matrix(sums, k);
            free(counts);
            return -1;
        }
    }

    /* מתחילים לרוץ על כל הנקודות במערכת*/
    for (i = 0; i < N; i++){
        /*מבררים לאיזה קלאסטר שייכת נקודה i*/
        cluster = nearest_centroid(points[i], centroids, K, dim);
        counts[cluster]++; /*מגדילים את הספירה של אותו קלאסטר*/
        /* מוסיפים את הנקודה לקלאסטר*/
        for (j = 0; j < dim; j++){
            sums[cluster][j] += points[i][j];
        }
    }

    /* שלב חישוב הממוצעים והזזת המרכז בהתאם*/
    for (k = 0; k < K; k++){
        change = 0.0;
        /* אם אף נקודה לא בקלאסטר, מדלגים עליה*/
        if (counts[k] == 0){
            continue;
        }
        /* מחשבים מחדש את המיקום החדש*/
        for (j = 0; j < dim; j++){
            new_value = sums[k][j] / counts[k];
            change += (centroids[k][j] - new_value) *
                      (centroids[k][j] - new_value);
            centroids[k][j] = new_value; /*מגדירים את המיקום החדש */
        }

        change = sqrt(change); /* בודקים את המרחק האוקלידי בצנטרואיד*/
        if (change >= epsilon){ 
            converged = 0;
        }
    }

    /* שחרור זיכרון*/
    free_matrix(sums, K);
    free(counts);
    return converged;
}

/* replaces main(). does not allocate or free memory, everything instead is passed as parameter and updated. */
/* status: -1 = allocation error, 0 = finished successfully, 1 = centroid convergence */
static int run_kmeans(double **points, double **centroids, int K, double epsilon,
                      int N, int dim, int max_iter) {
    int iteration;
    int status;

    for (iteration = 0; iteration < max_iter; iteration++) {
        status = update_centroids(points, centroids, N, K, dim, epsilon);
        if (status == -1) {
            return 1;
        }
        if (status == 1) {
            break;
        }
    }

    return 0;
}

/* converts a Python list of lists into a C double matrix. */
static double **python_to_matrix(PyObject *python_matrix, int rows, int cols) {
    double **matrix;
    PyObject *python_row;
    PyObject *python_item;
    int i, j;

    /* generate matrix */
    matrix = (double **)malloc(rows * sizeof(double *));
    if (matrix == NULL) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        return NULL;
    }

  
    for (i = 0; i < rows; i++) {
        /* obtain row from python object and check for mismatched dimensions */
        python_row = PyList_GetItem(python_matrix, i);
        if (python_row == NULL || PyObject_Length(python_row) != cols) 
        {
            PyErr_SetString(PyExc_ValueError, "An Error Has Occurred");
            free_matrix(matrix, i);
            return NULL;
        }

        /* generate columnss for row i */
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
            free_matrix(matrix, i);
            return NULL;
        }

        /* data entry/validation for vector i */
        for (j = 0; j < cols; j++) {
            python_item = PyList_GetItem(python_row, j); /* automatically sets Python error if it fails */
            if (python_item == NULL) {
                free_matrix(matrix, i + 1);
                return NULL; 
            }

            /* convert Python object into double and enter into matric */
            matrix[i][j] = PyFloat_AsDouble(python_item);
            if (PyErr_Occurred()) {
                free_matrix(matrix, i + 1);
                return NULL; 
            }
        }
    }

    return matrix;
}

/* converts matrix obtained from run_kmeans into a Python list of lists */
static PyObject *matrix_to_python(double **matrix, int rows, int cols) {
    PyObject *python_matrix;
    PyObject *python_row;
    PyObject *python_value;
    int i, j;

    python_matrix = PyList_New(rows);
    if (python_matrix == NULL) { /* memory allocation failure, Python error automatically set */
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        python_row = PyList_New(cols);
        if (python_row == NULL) {  /* memory allocation failure, Python error automatically set */
            Py_DECREF(python_matrix); /* frees partially created Python list. */
            return NULL;
        }

        for (j = 0; j < cols; j++) {
            /* converts C double to python float */
            python_value = Py_BuildValue("d", matrix[i][j]);
            if (python_value == NULL) {  /* conversion failure, Python error automatically set */
                Py_DECREF(python_row);
                Py_DECREF(python_matrix);
                return NULL;
            }
            PyList_SetItem(python_row, j, python_value);
        }

        PyList_SetItem(python_matrix, i, python_row);
    }

    return python_matrix;
}

/* Python facing wrapper */
static PyObject *fit(PyObject *self, PyObject *args) {
    PyObject *python_centroids;
    PyObject *python_points;
    PyObject *first_point;
    PyObject *result;
    double **points;
    double **centroids;
    double epsilon;
    int max_iter;
    int N, K, dim;

    (void)self; /* avoids compiler warning about unused self parameter, maybe unecessary */

    /* arguments passed in through python, "00id" tells C how to interpret each arg */
    if (!PyArg_ParseTuple(args, "OOid", &python_centroids, &python_points,
                          &max_iter, &epsilon)) {
        return NULL;
    }

    /* validates that the points, centroids are Python lists */
    if (!PyList_Check(python_points) || !PyList_Check(python_centroids)) {
        PyErr_SetString(PyExc_TypeError, "Points and centroids must be lists");
        return NULL;
    }

    /* Obtain N, K from Python list sizes */
    N = (int)PyObject_Length(python_points);
    K = (int)PyObject_Length(python_centroids);
    if (N <= 0 || K <= 0) {
        PyErr_SetString(PyExc_ValueError, "Points and centroids cannot be empty");
        return NULL;
    }

    first_point = PyList_GetItem(python_points, 0);
    if (first_point == NULL || !PyList_Check(first_point)) {
        PyErr_SetString(PyExc_TypeError, "Each point must be a list");
        return NULL;
    }

    /* obtain dimension size from first supplied python point */
    dim = (int)PyObject_Length(first_point);
    if (dim <= 0) {
        PyErr_SetString(PyExc_ValueError, "Points cannot be empty");
        return NULL;
    }

    /* prevents illegal values for K, max_iter, epsilon from reaching kmeans */
    if (K <= 1 || K >= N || max_iter <= 0 || epsilon < 0.0) {
        PyErr_SetString(PyExc_ValueError, "Invalid Kmeans parameters");
        return NULL;
    }

    /* generate C points matrix for kmeans */
    points = python_to_matrix(python_points, N, dim);
    if (points == NULL) {
        return NULL;
    }

    /* generate C centroids matrix for kmeans */
    centroids = python_to_matrix(python_centroids, K, dim);
    if (centroids == NULL) {
        free_matrix(points, N);
        return NULL;
    }

    /* run kmeans on obtained points, check for errors */
    if (run_kmeans(points, centroids, K, epsilon, N, dim, max_iter) != 0) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        free_matrix(points, N);
        free_matrix(centroids, K);
        return NULL;
    }

    /* return centroids to python lists */
    result = matrix_to_python(centroids, K, dim);
    free_matrix(points, N);
    free_matrix(centroids, K);

    return result;
}

/* Below are necessary components of any C extension: */

/* tells Python what methods are exposed */
static PyMethodDef kmeans_methods[] = {
    {
        "fit",
        (PyCFunction)fit,
        METH_VARARGS,
        PyDoc_STR("fit(centroids, points, max_iter, epsilon) runs kmeans using the supplied initial centroids and returns the final centroids.")
    },
    {NULL, NULL, 0, NULL}
};

/* module description */
static struct PyModuleDef kmeans_module = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    NULL,
    -1,
    kmeans_methods
};

/* called by Python when it imports mykmeanssp. */
PyMODINIT_FUNC PyInit_mykmeanssp(void) {
    PyObject *module;
    module = PyModule_Create(&kmeans_module);
    if (module == NULL) {
        return NULL;
    }
    return module;
}