#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "AutoencoderDocuments.h"
#include <stdarg.h>
#include <stdbool.h>

void produce_combined(
        int number_of_X_cols,
        unsigned int *X,
        int target_value,
        int accumulation,
		unsigned int *source_clauses,
		int source_rows,
		int source_columns,
		unsigned int *destination_clauses,
		int destination_rows,
		int destination_columns,
		int enable_log
)
{
	void store_clause_to_X(int index,int columns, unsigned int *clauses, int number_of_X_cols, unsigned int *X);
	FILE* file = fopen("result/output.txt", "a");
	if (file != NULL) {
		enable_printing = enable_log;
		myPrint(file, "\nStart new accumulation (%d) for target_value (%d) ",accumulation,target_value);
		// myPrint(file, "number_of_X_cols = %d\n",number_of_X_cols);
		// myPrint(file, "target_value = %d\n",target_value);
		// myPrint(file, "source_columns = %d\n",source_columns);
		// myPrint(file, "destination_columns = %d\n",destination_columns);
		// myPrint(file, "source_rows = %d\n",source_rows);
		// myPrint(file, "destination_rows = %d\n",destination_rows);
		
		enable_printing = enable_log;
		int row;
		int length_of_source = source_rows * source_columns;
		int length_of_destination = destination_rows * destination_columns;

		int number_of_features = number_of_X_cols;
		int number_of_literals = 2*number_of_features;

		unsigned int number_of_literal_chunks = (number_of_literals-1)/32 + 1;

		// Initialize example vector X
		memset(X, 0, number_of_literal_chunks * sizeof(unsigned int));
		for (int k = number_of_features; k < number_of_literals; ++k) {
			int chunk_nr = k / 32;
			int chunk_pos = k % 32;
			X[chunk_nr] |= (1U << chunk_pos);
		}
		
		if (source_columns == 0 || destination_columns == 0) {
			return;
		}

		if (target_value) {
			myPrint(file, "and selected clauses is ");
			for (int a = 0; a < accumulation; ++a) {
				int random_source_index = (rand() % source_rows);
				store_clause_to_X(random_source_index, source_columns, source_clauses,number_of_X_cols,X);
				myPrint(file, "S(%d) ",random_source_index);

				int random_destination_index = (rand() % destination_rows);
				store_clause_to_X(random_destination_index, destination_columns, destination_clauses,number_of_X_cols,X);
				myPrint(file, "D(%d) -- ",random_destination_index);
			}
		} else {
			myPrint(file, "and selected not exist fetures is ");
			int a = 0;
			while (a < accumulation) {
				int r = 0;
				int total_features = source_columns + destination_columns;
				while (r < total_features){
					int feature;
					bool featureExists;
					do {
						featureExists = false;
						feature = rand() % number_of_X_cols;
						for (int i = 0; i < length_of_source; i++) {
							if (source_clauses[i] == feature) {
								featureExists = true;
							}
						}
						for (int i = 0; i < length_of_destination; i++) {
							if (destination_clauses[i] == feature) {
								featureExists = true;
							}
						}
					} while (featureExists);
					myPrint(file, "F(%d) ",feature);
					store_feature_to_X(feature, X, number_of_X_cols);
					r++;
				}
				a++;
			}
		}
		fclose(file);
	}
}

void store_clause_to_X(int index,int columns, unsigned int *clauses, int number_of_X_cols, unsigned int *X){
	int start_index = index * columns;
	int end_index = (index + 1) * columns;
	for (int k = start_index; k < end_index; ++k) {
		if (clauses[k] > 0)
		{
        	store_feature_to_X(clauses[k], X, number_of_X_cols);
		}
    }
}
