#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define EPSILON 0.001
#define DEFAULT_ITER 400
#define MAX_LINE 10000

/* פונקציה שמוודאת את הארגומנטים*/
int parse_int(const char *str, int *out_val){
    double val;
    char *endptr = NULL;
    if (str == NULL || *str == '\0'){
        return 0;
}
val = strtod(str, &endptr);

if (endptr == str){
    return 0;
}

while (*endptr == ' ' || *endptr == '\t' || *endptr == '\r' || *endptr == '\n') {
    endptr++;
}

if (*endptr != '\0'){
    return 0;
}
if (floor(val) != val){
    return 0;
}
if (val < 1.0 || val > 2147483647.0){
    return 0;
}
*out_val = (int)val;
return 1;
}

/* חישוב המרחק האוקלידי בין שתי נקודות*/

double euclidean_distance(double *a, double *b, int dim) {
    double sum = 0.0;
    int i;
    for(i = 0; i < dim; i++){
        /* מכפילים איבר בעצמו במקום להשתמש בחזקה*/
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

/* רצה על הצנטרואידים ומחזיקה את האינדקס של הקרוב ביותר*/
int nearest_centroid(double *point, double **centroids, int K, int dim) {
    int k;
    int best = 0; /* מגדירים תחילה שהראשון הוא הכי קרוב*/
    double dist;
    /*מחשבים את המרחק לראשון ומגדירים כטוב ביותר*/
    double best_dist =  euclidean_distance(point, centroids[0], dim);
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

void free_matrix(double **matrix, int rows) {
    int i;
    /* אם המטריצה לא הוקצתה, לא עושה כלום*/
    if (matrix == NULL)
        return;
    
    /* עוברים שורה שורה ומשחררים את הקואורדינטות*/
    for (i = 0; i < rows; i++){
        if (matrix[i] != NULL){
            free(matrix[i]);
        }
    }

    /* משחררים את המטריצה הראשית*/
    free(matrix);
}

/* הפונקציה המרכזית לקריאת הקלט שלנו, הנקודות*/

double **read_points(int *N, int *dim) {
    double **points = NULL; /* המערך הראשי, מצביעים לנקודות*/
    double *point = NULL; /* בכל איטרציה, יעבוד עם נקודה בודדת*/
    double **tmp = NULL; /* מצביע זמני*/
    char line[MAX_LINE]; /* באפר שיכיל בכל פעם חלק מסוים מהטקסט*/
    char *ptr = NULL; /* מצביע עזר, שירוץ על התווים*/
    int first_line = 1; /* סימון בוליאני שאנחנו בשורה הראשונה.*/
    int i;

    /* קוראים שורה שורה, עד שמגיעים לסוף הקובץ*/
    while (fgets(line, sizeof(line), stdin) != NULL){
        if (line[0] == '\n' || line[0] == '\r'){
            continue;
        }

        /* מגלים את מימד המרחב, מהשורה הראשונה בלבד*/
        if (first_line){
            int commas = 0;
            for (i = 0; line[i] != '\0'; i++){
                if (line[i] == ','){
                    commas++;
                }
            }
           /* המימד הוא תמיד מספר הפסיקים ועוד אחד*/
            *dim = commas + 1;
            first_line = 0; /*משנים את הערך כדי לא לחזור בשורות הבאות*/
        }
        
        /* הקצאת זיכרון לנקודה שעובדים עליה*/
        point = (double *)malloc((*dim) * sizeof(double));
        if (point == NULL) {
            free_matrix(points, *N); /*יש קריסה או שסיימנו עם הנקודות, מפנים זיכרון*/
            return NULL;
        }

        /* הופכים את הטקסט לאוסף מספרים*/
        ptr = line;
        for (i = 0; i < *dim; i++){
            if (line[0] == '\n'){
                continue;
            }
            /* נעזרים בפונקציה להפוך את הטקסט לדאבל*/
            point[i] = strtod(ptr, &ptr);
            if (*ptr == ',') {
                ptr++; /*אם אנחנו בפסיק, מתקדמים למספר הבא*/
            }
        }

        tmp = (double **)realloc(points, (*N + 1) * sizeof(double *));
        if (tmp == NULL){
            free(point); /* משחררים את הנקודה הנוכחית*/
            free_matrix(points, *N); /*משחררים את המטריצה הישנה*/
            return NULL;
        }

        points = tmp; /*מעדכנים את המצביע לכתובת עם הנקודה החדשה*/
        points[*N] = point; /*שומרים את הנקודה החדשה, בתא אחרון במערך*/
        (*N)++; /* מקדמים את מונה הנקודות*/
    }
    
    return points; /* מחזירים את המטריצה המעודכנת*/
}

int update_centroids(double **points, double **centroids, int N, int K, int dim){
    double **sums;
    int *counts;
    int i, j, k;
    int cluster;
    int converged = 1;
    double change, new_value;

    /* מקצים טבלת סכומים*/
    sums = (double **)malloc(K * sizeof(double *));
    if (sums == NULL){
        return 0;
    }
    
    /* מקצים מערך מונים מאופס*/
    counts = (int *)calloc(K, sizeof(int));
    if (counts == NULL){
        free(sums);
        return 0;
    }

    for (k = 0; k < K; k++){
        sums[k] = (double *)calloc(dim, sizeof(double));
        if (sums[k] == NULL){
            free_matrix(sums, k);
            free(counts);
            return 0;
        }
    }

    /* מתחילים לרוץ על כל הנקודות במערכת*/
    for (i = 0; i < N; i++){
        /*מבררים לאיזה קלאסטר שייכת נקודה i*/
        cluster = nearest_centroid(points[i], centroids, K,dim);
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
        for (j = 0; j < dim; j ++){
            new_value = sums[k][j] / counts[k];
            change += (centroids[k][j] - new_value) * (centroids[k][j] - new_value);
            centroids[k][j] = new_value; /*מגדירים את המיקום החדש */
        }
    
        change = sqrt(change); /* בודקים את המרחק האוקלידי בצנטרואיד*/
        if (change >= EPSILON){
            converged = 0;
        }
    }

    /* שחרור זיכרון*/
    free_matrix(sums, K);
    free(counts);
    return converged; 
}

int main(int argc, char *argv[]) {

    /* חובה להצהיר בפורמט ANSI על המשתנים בתחילת הבלוק*/
    int K;
    int max_iter = DEFAULT_ITER;
    double **points = NULL; /*מערך דו מימדי של נקודות*/
    double **centroids = NULL; /*מערך דו מימדי של מרכזים*/
    int N = 0; /*סופר כמה נקודות נקראו*/
    int dim = 0; /*סופר את מספר המימדים*/
    int i, j, iteration;


    /* בדיקת כמות הארגומנטים שהועברו בטרמינל*/
    /* סופר גם את שם התוכנית עצמה*/
    if (argc != 2 && argc != 3){  
        printf("An Error Has Occurred\n");
        return 1;
    }
    
    /* מוודא שהמשתמש הזין מספר וממיר לאינטיגר*/
    if (!parse_int(argv[1], &K) || K <=1) {
        printf("Incorrect number of clusters!\n");
        return 1;
    }

    /* מוודא שמספר האיטרציות כראוי*/
    if (argc == 3){
        if (!parse_int(argv[2], &max_iter) || max_iter <= 1 || max_iter >= 800){
            printf("Incorrect maximum iteration!\n");
            return 1;
        }
    }

    points = read_points(&N, &dim);
    if(points == NULL){
        printf("An Error Has Occurred\n");
        return 1;
    }

    if(K >= N){
        printf("Incorrect number of clusters!\n");
        free_matrix(points, N);
        return 1;
    }

    centroids = calloc(K, sizeof(double *));
    if (centroids == NULL) {
        printf("An Error Has Occurred\n");
        free_matrix(points, N);
        return 1;
    }

    for (i = 0; i < K; i++){
        centroids[i] = (double *)malloc(dim * sizeof(double));
        if (centroids[i] == NULL) {
            printf("An Error Has Occurred\n");
            free_matrix(centroids, i);
            free_matrix(points, N);
            return 1;
        }
        for (j = 0; j < dim; j++){
            centroids[i][j] = points[i][j];
        }
    }

    for (iteration = 0; iteration < max_iter; iteration++) {
        if (update_centroids(points, centroids, N, K, dim)){
        break;
        }
    }
    
    for(i = 0; i < K; i++){
        for(j = 0; j < dim; j++){
            printf("%.4f", centroids[i][j]);
            if(j < dim - 1){
            printf(",");
        }
    }
    printf("\n");
}

    free_matrix(points, N);
    free_matrix(centroids, K);
    return 0;
}