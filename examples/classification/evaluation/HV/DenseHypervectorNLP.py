import os
import keras
import datetime
import numpy as np
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from tmu.models.classification.vanilla_classifier import TMClassifier
from scipy.sparse import lil_matrix

class HypervectorNLP:
    @staticmethod
    def get_processed_data(NUM_WORDS, INDEX_FROM):
        train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
        train_x,train_y = train
        test_x,test_y = test
        word_to_id = keras.datasets.imdb.get_word_index()
        word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        return train_x,train_y,test_x,test_y

    @staticmethod
    def encode_input_text(hypervector_size, train_x, train_y, test_x, test_y, encoding):
        X_train = np.zeros((train_y.shape[0], hypervector_size), dtype=np.int32)
        Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
        for i in range(train_y.shape[0]):
            seen = {}
            for word_id in train_x[i]:
                if word_id not in seen:
                    X_train[i] += encoding[word_id]
                    seen[word_id] = True
            Y_train[i] = train_y[i]
        X_train = np.where(X_train >= 0, 1, 0).astype(np.uint32)

        X_test = np.zeros((test_y.shape[0], hypervector_size), dtype=np.int32)
        Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)
        for i in range(test_y.shape[0]):
            seen = {}
            for word_id in test_x[i]:
                if word_id not in seen:
                    X_test[i] += encoding[word_id]
                    seen[word_id] = True
            Y_test[i] = test_y[i]
        X_test = np.where(X_test >= 0, 1, 0).astype(np.uint32)
        return X_train,Y_train,X_test,Y_test

   
    @staticmethod
    def get_test_name(title):
        current_time = datetime.datetime.now()
        test_id = current_time.strftime("%Y%m%d%H%M%S")
        result_filename = f"{title}_{test_id}"
        result_filepath = os.path.join("tests" , result_filename + '.txt')
        return result_filepath
    
    @staticmethod
    def apply_tsetlin_machine(clauses, T, s, weighted_clauses, clause_drop_p, X_train, Y_train, X_test, Y_test):
        tm = TMClassifier(clauses, T, s, platform='CUDA', weighted_clauses=weighted_clauses, clause_drop_p=clause_drop_p)
        print("Accuracy over 40 epochs:")
        for i in range(40):
            start_training = time()
            tm.fit(X_train, Y_train)
            stop_training = time()

            start_testing = time()
            result = 100*(tm.predict(X_test) == Y_test).mean()
            stop_testing = time()

            print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))