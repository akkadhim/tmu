extern int enable_printing;

void myPrint(
    FILE *file, 
    const char* format, 
    ...
);

int generate_random(
    int rows
);

unsigned int compareints(
    const void * a, 
    const void * b
);

unsigned int compareIndexedValues(
    const void *a, 
    const void *b
);

void store_feature_to_X(
    int feature, 
    int number_of_cols, 
    unsigned int *X, 
    int output_pos
);
