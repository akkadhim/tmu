#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "AutoencoderDocuments.h"
#include <stdarg.h>
#include <stdbool.h>

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

typedef struct {
    unsigned int value;
    unsigned int index;
} IndexedValue;

unsigned int compareints(const void * a, const void * b)
{
  return(*(unsigned int*)a- *(unsigned int*)b);
}

// Comparison function for sorting IndexedValue array
unsigned int compareIndexedValues(const void *a, const void *b) {
    return ((IndexedValue *)a)->value - ((IndexedValue *)b)->value;
}