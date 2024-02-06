import os
import pickle
import numpy as np
import datetime
from collections import defaultdict
from knowledge import Knowledge

target_similarity=defaultdict(list)
from contextlib import redirect_stdout
from tqdm import tqdm

def preprocess_text(text):
    return text

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

f_vectorizer_X = open("vectorizer_X.pickle", "rb")
vectorizer_X = pickle.load(f_vectorizer_X)
f_vectorizer_X.close()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]

f_X = open("X.pickle", "rb")
X_train = pickle.load(f_X)
f_X.close()

folder_path = 'datasets'
for folder_name in os.listdir(folder_path):
    if folder_name == 'rg-65':
        current_folder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(current_folder_path):
            files_start_name = os.path.join(current_folder_path, folder_name)
            with open(files_start_name + '_word1.pkl', 'rb') as f:
                word1 = pickle.load(f)
            with open(files_start_name + '_word2.pkl', 'rb') as f:
                word2 = pickle.load(f)
            word_total= list(set(word1 + word2))

            target_words=[]
            for i in word_total:
                if i in vectorizer_X.vocabulary_:
                    target_words.append(i)
            output= open(files_start_name + '_target.pkl', "wb")
            pickle.dump(target_words, output)
            output.close()

            output_active = np.empty(len(target_words), dtype=np.uint32)
            for i in range(len(target_words)):
                target_word = target_words[i]
                target_id = vectorizer_X.vocabulary_[target_word]
                output_active[i] = target_id

            current_time = datetime.datetime.now()
            test_id = current_time.strftime("%Y%m%d%H%M%S")
            result_filename = f"{folder_name}_phase1_{test_id}"
            test_start_name = os.path.join(current_folder_path, "tests")
            result_filepath = os.path.join(test_start_name , result_filename + '.txt')

            with open(result_filepath, 'w') as file, redirect_stdout(file):
                print("Loading dataset: " + folder_name)
                print("")
                print("Epochs: %d" % epochs)
                print("Example: %d" % number_of_examples)
                print("Target words: %d" % len(target_words))
                print("Accumulation: %d" % accumulation)
                print("No of features: %d" % number_of_features)
                output_active_list = output_active
                total_training_time = 0
                knowledge_start_name = os.path.join(current_folder_path, "knowledge")

                words_progress_bar = tqdm(total=len(output_active), desc="Running Words")
                for i in output_active_list:
                    knowledge_filepath = os.path.join(knowledge_start_name , str(i) + '.pkl')
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
                            knowledge_filepath = os.path.join(knowledge_start_name , str(literal) + '.pkl')
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
                seconds = total_training_time
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60
                print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")