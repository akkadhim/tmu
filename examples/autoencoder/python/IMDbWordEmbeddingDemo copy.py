import datetime
import keras
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import CountVectorizer
from time import time
# from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from contextlib import redirect_stdout
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
target_words = [
    'apple','orange'
]
home_dir = os.path.expanduser("~")
root_folder = os.path.join(home_dir, "tmu_results")
if not os.path.exists(root_folder):
    os.makedirs(root_folder)

combined = True
experts_dataset = []
involved_datasets = []

if combined == True:
    involved_datasets.append(["Combined",0,1])\
	
NUM_WORDS=10000
INDEX_FROM=2
train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
train_x,train_y = train
print("the number of reviews in train set =",train_x.size)
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id.pop("<PAD>", None)
word_to_id.pop("<START>", None)
word_to_id.pop("<UNK>", None)
id_to_word = {value:key for key,value in word_to_id.items()}
for i in range(train_y.shape[0]):
	terms = []
	for word_id in train_x[i]:
		if word_id in id_to_word:
			terms.append(id_to_word[word_id].lower())             
	experts_dataset.append(terms)

old_length = len(experts_dataset)
if combined == False:
    involved_datasets.append(["IMDB",old_length,len(experts_dataset)])
    
def tokenizer(s):
    return s
vectorizer_X = CountVectorizer(tokenizer=tokenizer, lowercase=False, binary = True)
X_train = vectorizer_X.fit_transform(experts_dataset)
feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
print("No of features: %d" % number_of_features)
output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
    target_word = target_words[i]
    if target_word in vectorizer_X.vocabulary_:
        target_id = vectorizer_X.vocabulary_[target_word]
        output_active[i] = target_id
    else:
        print(f"Warning: '{target_word}' not found in vocabulary.")
print("tokenizing target words completed")

from scipy.sparse import csr_matrix
X_train_array = X_train.toarray()
first_row = X_train_array[1]

# Set the numpy print options to display all elements
np.set_printoptions(threshold=np.inf)
print(first_row)

file_path = "result/output.txt"
if os.path.exists(file_path):
    os.remove(file_path)
current_time = datetime.datetime.now()
test_id = current_time.strftime("%Y%m%d%H%M%S")
result_filename = f"result_{test_id}.txt"
test_dir = os.path.join(root_folder, test_id)
os.makedirs(test_dir)
result_filepath = os.path.join(test_dir, result_filename)
clauses_dir = os.path.join(test_dir, 'clauses_variance_plots')
if not os.path.exists(clauses_dir):
    os.makedirs(clauses_dir)

# parameters
clause_weight_threshold = 0
number_of_examples = 4000
factor = 4
T = factor*40
s = 5.0
#clauses = factor*5
clauses = 20
clause_increment = False
random_per_category = False
categories = 0
accumulation = 24
epochs = 2
fluctuations = []
progress_bar = tqdm(total=epochs, desc="Running Epochs")
# combined
target_words_clauses = []
    
with open(result_filepath, 'w') as file, redirect_stdout(file):
    print("Test: %s" % test_id)

    class_index = np.arange(len(output_active), dtype=np.uint32)
    for i in output_active:
        target_word_clauses = []

        shape = (1, 1)
        single_output_active = np.empty(1, dtype=np.uint32)
        single_output_active[0] = i
        tm = TMAutoEncoder(clauses, T, s, single_output_active, max_included_literals=3, accumulation=accumulation, feature_negation=False, platform='CPU', output_balancing=True)
        total_training = 0
        if(categories > 0): 
            print("Algorithm: With %d categories and random per category" % categories) if(random_per_category) else print("Algorithm: With %d categories" % categories) 
        else: 
            print("Algorithm: Original without categories")
        print("Epochs: %d" % epochs)
        print("Example: %d" % number_of_examples)
        print("Target words: %d" % len(target_words))
        print("Accumulation: %d" % accumulation)
        print("Datasets involved: %s" % involved_datasets)
        print("No of features: %d" % number_of_features)
        print("Clauses: %d with increment by 2\n" % clauses) if clause_increment else print("Clauses: %d\n" % clauses)
        
        for e in range(epochs):
            print("\nEpoch #%d" % (e+1))
            start_training = time()
            if categories > 0:
                tm.fit(
                    X_train, 
                    number_of_examples=number_of_examples, 
                    categories=categories, 
                    random_per_category = random_per_category,
                    involved_datasets=involved_datasets 
                    )
            else:
                tm.fit(
                    X_train, 
                    number_of_examples=number_of_examples, 
                    involved_datasets=involved_datasets 
                    )
            stop_training = time()
            total_training = total_training + (stop_training - start_training)

            
            if((e+1) == epochs):
                print("\n=====================================\nClauses\n=====================================")
                for j in range(clauses):
                    print("Clause #%-2d " % (j), end=' ')
                    l = [] 
                    related_literals = []
                    number_of_literals = 0 
                    for k in range(tm.clause_bank.number_of_literals):
                        if tm.get_ta_action(j, k) == 1:
                            number_of_literals = number_of_literals + 1
                            if k < tm.clause_bank.number_of_features:
                                related_literals.append(k)
                                l.append("%s(%d)" % (feature_names[k], tm.clause_bank.get_ta_state(j, k)))
                            else:
                                l.append("¬%s(%d)" % (feature_names[k-tm.clause_bank.number_of_features], tm.clause_bank.get_ta_state(j, k)))
                    print(": No of features:%-6d" % (number_of_literals), end=" ==> ")
                    try:
                        print(" - ".join(l).encode('utf-8', errors='ignore'))
                    except UnicodeEncodeError:
                        print(" exception ")
                    target_word_clauses.append([related_literals])
                print(target_word_clauses)
            progress_bar.update(1)
        target_words_clauses.append([i,target_word_clauses])

    seconds = total_training
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")
progress_bar.close()