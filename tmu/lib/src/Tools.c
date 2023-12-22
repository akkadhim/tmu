#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include "Tools.h"

int enable_printing = 0;

void myPrint(FILE *file, const char* format, ...)
{
	if (enable_printing == 1)
	{
    	va_list args;
		va_start(args, format);
		vfprintf(file, format, args);

		va_end(args);
	}
}

int generate_random(int rows){
	return rand() % rows;
}

unsigned int compareints(const void * a, const void * b)
{
  return(*(unsigned int*)a- *(unsigned int*)b);
}

void store_feature_to_X(int feature, int number_of_cols, unsigned int *X, int output_pos)
{
    int chunk_nr = feature / 32;
    int chunk_pos = feature % 32;
    X[output_pos + chunk_nr] |= (1U << chunk_pos);

    chunk_nr = (feature + number_of_cols) / 32;
    chunk_pos = (feature + number_of_cols) % 32;
    X[output_pos + chunk_nr] &= ~(1U << chunk_pos);
}