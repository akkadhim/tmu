#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "AutoencoderClauses.h"
#include "Tools.h"
#include <stdarg.h>
#include <stdbool.h>

void store_clause_to_X(int index,int columns, unsigned int *clauses, int number_of_features, unsigned int *X){
	int start_index = index * columns;
	int end_index = (index + 1) * columns;
	for (int k = start_index; k < end_index; ++k) {
		int feature = clauses[k];
		if (feature > 0 && feature < number_of_features)
		{
			store_feature_to_X(feature, number_of_features, X, 0);
		}
    }
}

void produce_example_by_combined_clauses(
        int number_of_cols,
        unsigned int *X,
        int target_value,
        int accumulation,
		unsigned int *source_clauses,
		int *source_clauses_weights,
		int source_rows,
		int source_columns,
		unsigned int *destination_clauses,
		int *destination_clauses_weights,
		int destination_rows,
		int destination_columns,
		int negative_weight_clause,
		int enable_log
)
{
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
		
		int row;
		int length_of_source = source_rows * source_columns;
		int length_of_destination = destination_rows * destination_columns;

		int number_of_features = number_of_cols;
		int number_of_literals = 2*number_of_features;

		unsigned int number_of_literal_chunks = (number_of_literals-1)/32 + 1;

		// Initialize example vector X
		memset(X, 0, number_of_literal_chunks * sizeof(unsigned int));
		for (int k = number_of_features; k < number_of_literals; ++k) {
			int chunk_nr = k / 32;
			int chunk_pos = k % 32;
			X[chunk_nr] |= (1U << chunk_pos);
		}
		
		if (source_columns == 0 || destination_columns == 0 || source_rows == 0 || destination_rows == 0) {
			return;
		}

		if (target_value) {
			myPrint(file, "and selected positive clauses is ");
			int a = 0;
			while (a < accumulation) {
				bool positive_clause = false;

				int random_source_index = (rand() % source_rows);
				if (source_clauses_weights[random_source_index] > 0)
				{
					store_clause_to_X(random_source_index, source_columns, source_clauses,number_of_cols,X);
					int random_destination_index = (rand() % destination_rows);
					if (destination_clauses_weights[random_destination_index] > 0)
					{
						store_clause_to_X(random_destination_index, destination_columns, destination_clauses,number_of_cols,X);
						
						myPrint(file, "S(%d) ",random_source_index);
						myPrint(file, "D(%d) -- ",random_destination_index);
						positive_clause = true;
					}
				}
				if (positive_clause)
				{
					a++;
				}
			}
		} else {
			myPrint(file, "and selected negative clauses is ");
			if(negative_weight_clause){
				int a = 0;
				while (a < accumulation) {
					bool negative_clause = false;

					int random_source_index = (rand() % source_rows);
					if (source_clauses_weights[random_source_index] < 0)
					{
						store_clause_to_X(random_source_index, source_columns, source_clauses,number_of_cols,X);
						int random_destination_index = (rand() % destination_rows);
						if (destination_clauses_weights[random_destination_index] < 0)
						{
							store_clause_to_X(random_destination_index, destination_columns, destination_clauses,number_of_cols,X);
							
							myPrint(file, "S(%d) ",random_source_index);
							myPrint(file, "D(%d) -- ",random_destination_index);
							negative_clause = true;
						}
					}
					if (negative_clause)
					{
						a++;
					}
				}
			}
			else{
				int a = 0;
				while (a < accumulation) {
					int r = 0;
					int total_features = source_columns + destination_columns;
					while (r < total_features){
						int feature;
						bool featureExists;
						do {
							featureExists = false;
							feature = rand() % number_of_cols;
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
						store_feature_to_X(feature, number_of_cols, X,0);
						r++;
					}
					a++;
				}
			}
		}
		fclose(file);
	}
}

void produce_example_by_clauses(
        int number_of_features,
        unsigned int *X,
        int target_value,
        int accumulation,
		unsigned int *source_clauses,
		int *source_clauses_weights,
		int source_rows,
		int source_columns,
		int negative_weight_clause,
		int enable_log
)
{
	FILE* file = fopen("result/output.txt", "a");
	if (file != NULL) {
		enable_printing = enable_log;
		myPrint(file, "\nStart new accumulation (%d) for target_value (%d) ",accumulation,target_value);
		
		int row;
		int length_of_source = source_rows * source_columns;
		int number_of_literals = 2*number_of_features;

		unsigned int number_of_literal_chunks = (number_of_literals-1)/32 + 1;

		// Initialize example vector X
		memset(X, 0, number_of_literal_chunks * sizeof(unsigned int));
		for (int k = number_of_features; k < number_of_literals; ++k) {
			int chunk_nr = k / 32;
			int chunk_pos = k % 32;
			X[chunk_nr] |= (1U << chunk_pos);
		}
		
		if (source_columns == 0 || source_rows == 0) {
			return;
		}

		if (target_value) {
			myPrint(file, "and selected positive clauses is ");
			for (int i = 0; i < source_rows; i++)
			{
				if (source_clauses_weights[i] > 0)
				{
					store_clause_to_X(i, source_columns, source_clauses,number_of_features,X);
					myPrint(file, "S(%d) ",i);
				}
			}
			
			// int a = 0;
			// while (a < accumulation) {
			// 	bool positive_clause = false;

			// 	int random_source_index = (rand() % source_rows);
			// 	if (source_clauses_weights[random_source_index] > 0)
			// 	{
			// 		store_clause_to_X(random_source_index, source_columns, source_clauses,number_of_features,X);
			// 		myPrint(file, "S(%d) ",random_source_index);
			// 		positive_clause = true;
			// 	}

			// 	if (positive_clause)
			// 	{
			// 		a++;
			// 	}
			// }
		} else {
			myPrint(file, "and selected negative clauses is ");
			if(negative_weight_clause){
				for (int i = 0; i < source_rows; i++)
				{
					if (source_clauses_weights[i] < 0)
					{
						store_clause_to_X(i, source_columns, source_clauses,number_of_features,X);
						myPrint(file, "S(%d) ",i);
					}
				}

				// int a = 0;
				// while (a < accumulation) {
				// 	bool negative_clause = false;

				// 	int random_source_index = (rand() % source_rows);
				// 	if (source_clauses_weights[random_source_index] < 0)
				// 	{
				// 		store_clause_to_X(random_source_index, source_columns, source_clauses,number_of_features,X);
				// 		myPrint(file, "S(%d) ",random_source_index);
				// 		negative_clause = true;
				// 	}
				// 	if (negative_clause)
				// 	{
				// 		a++;
				// 	}
				// }
			}
			else{
				int a = 0;
				while (a < accumulation) {
					int r = 0;
					int total_features = source_columns;
					while (r < total_features){
						int feature;
						bool featureExists;
						do {
							featureExists = false;
							feature = rand() % number_of_features;
							for (int i = 0; i < length_of_source; i++) {
								if (source_clauses[i] == feature) {
									featureExists = true;
								}
							}
						} while (featureExists);
						myPrint(file, "F(%d) ",feature);
						store_feature_to_X(feature, number_of_features, X,0);
						r++;
					}
					a++;
				}
			}
		}
		fclose(file);
	}
}
