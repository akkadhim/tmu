
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
);

void produce_example_by_clauses(
        int number_of_cols,
        unsigned int *X,
        int target_value,
        int accumulation,
		unsigned int *clauses,
		int *clauses_weights,
		int rows,
		int columns,
		int negative_weight_clause,
		int enable_log
);
