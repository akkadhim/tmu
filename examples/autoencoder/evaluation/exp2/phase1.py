import os
import pickle
from tqdm import tqdm
from collections import defaultdict
from contextlib import redirect_stdout
from knowledge import Knowledge
from tools import Tools
from directories import Dicrectories

target_similarity=defaultdict(list)
clause_weight_threshold = 10
number_of_examples = 2000
accumulation = 25
clause_drop_p = 0.0
factor = 20
clauses = int(factor*20/(1.0 - clause_drop_p))
T = factor*40
s = 5.0
epochs = 25

knowledge = Knowledge(
    clause_weight_threshold, 
    number_of_examples, 
    accumulation, 
    clause_drop_p, 
    factor, 
    T, 
    s, 
    epochs)

def preprocess_text(text):
    return text
vectorizer_X = Tools.read_pickle_data("vectorizer_X.pickle")
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
X_train = Tools.read_pickle_data("X.pickle")

for dataset_name in os.listdir(Dicrectories.datasets):
    if dataset_name == 'rg-65':
        current_folder_path = os.path.join(Dicrectories.datasets, dataset_name)
        if os.path.isdir(current_folder_path):
            files_start_name = os.path.join(current_folder_path, dataset_name)
            output_active, target_words = Tools.generate_targets(files_start_name)
            knowledge_directory = Dicrectories.knowledge(dataset_name)
            
            result_filepath = Dicrectories.test(dataset_name,"phase1")
            with open(result_filepath, 'w') as file, redirect_stdout(file):
                print("Loading dataset: " + dataset_name)
                print("")
                print("Epochs: %d" % epochs)
                print("Example: %d" % number_of_examples)
                print("Target words: %d" % len(target_words))
                print("Accumulation: %d" % accumulation)
                print("No of features: %d" % number_of_features)
                output_active_list = output_active
                total_training_time = 0

                words_progress_bar = tqdm(total=len(output_active), desc="Running Words")
                for i in output_active_list:
                    knowledge_filepath = os.path.join(knowledge_directory , str(i) + '.pkl')
                    if os.path.exists(knowledge_filepath):
                        print("\nTW file exists: %s" % vectorizer_X.get_feature_names_out()[i])
                        with open(knowledge_filepath, 'rb') as f:
                            target_word_clauses = pickle.load(f)
                        training_time = 0
                    else:
                        print("\nTW run: %s" % vectorizer_X.get_feature_names_out()[i])
                        training_time, target_word_clauses = knowledge.generate(X_train, current_folder_path, i)
                        
                    total_training_time = total_training_time + training_time
                    clauses_progress_bar = tqdm(total=len(target_word_clauses), desc="Running Clauses")
                    for clause in target_word_clauses:
                        # weight = clause[0]
                        related_literals = clause[1]
                        feature_progress_bar = tqdm(total=len(related_literals), desc="Running Features")
                        for literal in related_literals:
                            knowledge_filepath = os.path.join(knowledge_directory , str(literal) + '.pkl')
                            if os.path.exists(knowledge_filepath):
                                print("Feature file exists: %s" % vectorizer_X.get_feature_names_out()[literal])
                            else:
                                print("Feature run: %s" % vectorizer_X.get_feature_names_out()[literal])
                                training_time, inner_target_word_clauses = knowledge.generate(X_train, current_folder_path, literal)
                                total_training_time = total_training_time + training_time
                            feature_progress_bar.update(1)
                        feature_progress_bar.close()
                        clauses_progress_bar.update(1)
                    clauses_progress_bar.close()

                    words_progress_bar.update(1)
                words_progress_bar.close()                
                Tools.print_training_time(total_training_time)