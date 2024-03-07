import os
import keras
import datetime
import numpy as np
from time import time
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from tmu.models.classification.vanilla_classifier import TMClassifier
from scipy.sparse import lil_matrix

class HypervectorNLP:
    @staticmethod
    def generate_encoding_hypervector(hypervector_size, bits, FEATURES):
        # print("Producing encoding...")
        encoding = np.zeros((FEATURES, hypervector_size), dtype=np.uint32)
        #from 0 tp HV size
        indexes = np.arange(hypervector_size, dtype=np.uint32)
        for i in range(FEATURES):
            selection = np.random.choice(indexes, size=(bits))
            encoding[i][selection] = 1

        # hash_value =  np.zeros(hypervector_size, dtype=np.uint32)
        # for i in range(FEATURES):
        #     one_ids = encoding[i].nonzero()[0]
        #     hash_value[i] = hash(one_ids.tobytes()) % (2 ** 29 - 1)
        return encoding

    @staticmethod
    def get_processed_data(NUM_WORDS, INDEX_FROM):
        # print("Downloading dataset...")
        train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
        train_x,train_y = train
        test_x,test_y = test

        word_to_id = keras.datasets.imdb.get_word_index()
        word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value:key for key,value in word_to_id.items()}

        # print("Producing bit representation...")
        training_documents = []
        for i in range(train_y.shape[0]):
            terms = []
            for word_id in train_x[i]:
                terms.append(id_to_word[word_id].lower())
            training_documents.append(terms)

        testing_documents = []
        for i in range(test_y.shape[0]):
            terms = []
            for word_id in test_x[i]:
                terms.append(id_to_word[word_id].lower())
            testing_documents.append(terms)

        def tokenizer(s):
            return s

        vectorizer_X = CountVectorizer(tokenizer=tokenizer, lowercase=False, ngram_range=(1,2), binary=True)

        X_train = vectorizer_X.fit_transform(training_documents)
        X_test = vectorizer_X.transform(testing_documents)
        return train_y,test_y,X_train,X_test

    @staticmethod
    def apply_feature_selection(FEATURES, train_y, X_train, X_test):
        # print("Selecting features...")

        SKB = SelectKBest(chi2, k=FEATURES)
        SKB.fit(X_train, train_y)

        X_train_org = SKB.transform(X_train)
        X_test_org = SKB.transform(X_test)
        return X_train_org,X_test_org
    
    @staticmethod
    def encode_input_text(hypervector_size, encoding, train_y, test_y, X_train_org, X_test_org):
        X_train = np.zeros((train_y.shape[0], hypervector_size), dtype=np.uint32)
        Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
        for i in range(train_y.shape[0]):
            #get ith document and then call it nonzeros features and loop over them
            #for each feature integrate it's relvant features from HV in ith document 
            for word_id in X_train_org.getrow(i).indices:
                X_train[i] = np.logical_or(X_train[i], encoding[word_id])
            Y_train[i] = train_y[i]

        X_test = np.zeros((test_y.shape[0], hypervector_size), dtype=np.uint32)
        Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)
        for i in range(test_y.shape[0]):
            for word_id in X_test_org.getrow(i).indices:
                X_test[i] = np.logical_or(X_test[i], encoding[word_id])
            Y_test[i] = test_y[i]
        return X_train,X_test,Y_train,Y_test
    
    @staticmethod
    def encode_arrays(train_y, test_y, X_train_org, X_test_org):
        X_train = X_train_org.toarray()
        Y_train = np.zeros(train_y.shape[0], dtype=np.uint32)
        for i in range(train_y.shape[0]):
            Y_train[i] = train_y[i]

        X_test = X_test_org.toarray()
        Y_test = np.zeros(test_y.shape[0], dtype=np.uint32)
        for i in range(test_y.shape[0]):
            Y_test[i] = test_y[i]
        return X_train,X_test,Y_train,Y_test
    
    @staticmethod
    def get_test_name(title):
        current_time = datetime.datetime.now()
        test_id = current_time.strftime("%Y%m%d%H%M%S")
        result_filename = f"{title}_{test_id}"
        result_filepath = os.path.join("tests" , result_filename + '.txt')
        return result_filepath
    
    @staticmethod
    def apply_simulated_annealing(clauses, T, s, X_train, X_test, Y_train, Y_test):
        tm = TMClassifier(clauses, T, s, platform='CUDA')
        print("\nAccuracy over 40 epochs:\n")
        for i in range(40):
            start_training = time()
            tm.fit(X_train, Y_train)
            stop_training = time()

            start_testing = time()
            result = 100*(tm.predict(X_test) == Y_test).mean()
            stop_testing = time()

            print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, round(stop_training-start_training, 2), round(stop_testing-start_testing, 2)))

