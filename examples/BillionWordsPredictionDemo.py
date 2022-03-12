import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tmu.tsetlin_machine import TMClassifier
import pickle

target_word = 'awful'#'comedy'#'romance'#"scary"

examples = 10000
context_size = 25
profile_size = 50

clauses = 10
T = 40
s = 5.0

NUM_WORDS=10000
INDEX_FROM=2 

# Data obtained from https://www.kaggle.com/c/billion-word-imputation

#f = open("train_v2.txt")
#sentences = f.read().split("\n")
#f.close()

#vectorizer_X = CountVectorizer(max_features=NUM_WORDS, binary=True)
#X = vectorizer_X.fit_transform(sentences)

#f_vectorizer_X = open("vectorizer_X.pickle", "wb")
#pickle.dump(vectorizer_X, f_vectorizer_X, protocol=4)
#f_vectorizer_X.close()

print("Loading Vectorizer")
f_vectorizer_X = open("vectorizer_X.pickle", "rb")
vectorizer_X = pickle.load(f_vectorizer_X)
f_vectorizer_X.close()

#f_X = open("X.pickle", "wb")
#pickle.dump(X, f_X, protocol=4)
#f_X.close()

print("Loading Data")
f_X = open("X.pickle", "rb")
X_csr = pickle.load(f_X)
f_X.close()

X_csc = X_csr.tocsc()

feature_names = vectorizer_X.get_feature_names_out()
number_of_features = vectorizer_X.get_feature_names_out().shape[0]
target_id = vectorizer_X.vocabulary_[target_word]
Y = X_csc[:,target_id].toarray().reshape(X_csc.shape[0])
cols = np.arange(number_of_features) != target_id
X_csc = X_csc[:,cols] 

X_csr = X_csc.tocsr()

X_train, X_test, Y_train, Y_test = train_test_split(X_csr, Y, test_size=0.5)

print("Creating Contexts")

X_train_0 = X_train[Y_train==0]
Y_train_0 = Y_train[Y_train==0]
X_train_1 = X_train[Y_train==1]
Y_train_1 = Y_train[Y_train==1]

print("Number of Target Words:", Y_train_1.shape[0])

X_train = np.zeros((examples, number_of_features-1), dtype=np.uint32)
Y_train = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	if np.random.rand() <= 0.5:
		for c in range(context_size):
			X_train[i] = np.logical_or(X_train[i], X_train_1[np.random.randint(X_train_1.shape[0]),:].toarray())
		Y_train[i] = 1
	else:
		#for c in range(context_size):
		X_train[i] = np.logical_or(X_train[i], X_train_0[np.random.randint(X_train_0.shape[0]),:].toarray())
		Y_train[i] = 0

X_test_0 = X_test[Y_test==0]
Y_test_0 = Y_test[Y_test==0]
X_test_1 = X_test[Y_test==1]
Y_test_1 = Y_test[Y_test==1]
X_test = np.zeros((examples, number_of_features-1), dtype=np.uint32)
Y_test = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	if np.random.rand() <= 0.5:
		for c in range(context_size):
			X_test[i] = np.logical_or(X_test[i], X_test_1[np.random.randint(X_test_1.shape[0])].toarray())
		Y_test[i] = 1
	else:
		#for c in range(context_size):
		X_test[i] = np.logical_or(X_test[i], X_test_0[np.random.randint(X_test_0.shape[0])].toarray())
		Y_test[i] = 0

tm = TMClassifier(clauses, T, s, platform='CPU', weighted_clauses=True)

print("\nAccuracy Over 40 Epochs:\n")
for i in range(40):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("\nPositive Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		if k < number_of_features:
			print(feature_names[k], end=' ')
		else:
			print("¬" + feature_names[k - number_of_features], end=' ')

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=False).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		if k < number_of_features:
			print(feature_names[k], end=' ')
		else:
			print("¬" + feature_names[k - number_of_features], end=' ')
	print()

	print("\nNegative Polarity:", end=' ')
	literal_importance = tm.literal_importance(1, negated_features=False, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		if k < number_of_features:
			print(feature_names[k], end=' ')
		else:
			print("¬" + feature_names[k - number_of_features], end=' ')

	literal_importance = tm.literal_importance(1, negated_features=True, negative_polarity=True).astype(np.int32)
	sorted_literals = np.argsort(-1*literal_importance)[0:profile_size]
	for k in sorted_literals:
		if literal_importance[k] == 0:
			break

		if k < number_of_features:
			print(feature_names[k], end=' ')
		else:
			print("¬" + feature_names[k - number_of_features], end=' ')
	print()

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
