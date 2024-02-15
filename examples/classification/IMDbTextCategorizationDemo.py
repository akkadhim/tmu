import argparse
import logging
import numpy as np
import keras
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
import random
from directories import Dicrectories
from tools import Tools

_LOGGER = logging.getLogger(__name__)

def get_imdb(_LOGGER, args):
    _LOGGER.info("Preparing dataset")
    train, test = keras.datasets.imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)
    train_x, train_y = train
    test_x, test_y = test

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    _LOGGER.info("Preparing dataset.... Done!")

    _LOGGER.info("Producing bit representation...")

    id_to_word = {value: key for key, value in word_to_id.items()}

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

    vectorizer_X = CountVectorizer(
        tokenizer=lambda s: s,
        token_pattern=None,
        ngram_range=(1, args.max_ngram),
        lowercase=False,
        binary=True
    )

    X_train = vectorizer_X.fit_transform(training_documents)
    Y_train = train_y.astype(np.uint32)

    X_test = vectorizer_X.transform(testing_documents)
    Y_test = test_y.astype(np.uint32)
    _LOGGER.info("Producing bit representation... Done!")

    _LOGGER.info("Selecting Features....")

    SKB = SelectKBest(chi2, k=args.features)
    SKB.fit(X_train, Y_train)

    selected_features = SKB.get_support(indices=True)
    X_train = SKB.transform(X_train).toarray()
    X_test = SKB.transform(X_test).toarray()
    return X_train,Y_train,X_test,Y_test

def store_to_X(clause, X):
    for feature in clause:
        chunk_nr = feature // 32
        chunk_pos = feature % 32
        X[chunk_nr] |= (1 << chunk_pos)
        
def generate_knowledge(number_of_documents, args):
    #each document is set of features and has class either 0 or 1
    #context of document has vote or in case of imdb a user review is either +ve or -ve
    #how to build the document from knowledge
    #for each tw get its +ve and -ve clauses
    #from the +ve clauses take the most heigh and then get thier features. This is a class 1 document
    #from the selected features get also the most heigh clauses
    #start accumlating the features to form a document and store them along target value 1 if +ve and 0 if -ve
    #take part for training and part for test
    #feed the knowledge to the classifer TM
    #now I have to decide: 1-how many clauses to pick 2-how much for training and testing
    accumulation = 30
    documents_X = []
    documents_Y = []
    number_of_ta_chunks = int((args.features - 1) / 32 + 1)
    
    file_list = Dicrectories.get_all_knowledge_files()

    for i in range(number_of_documents):
        X = np.ascontiguousarray(np.zeros(number_of_ta_chunks, dtype=np.uint32))

        random_file = random.choice(file_list)
        tw_knowledge_path = Dicrectories.get_knowledge_file(random_file)
        tw_all_clauses = Tools.read_pickle_data(tw_knowledge_path)

        target_value = random.randint(0, 1)
        if target_value == 1:
            tw_filtered_clauses = [clause for clause in tw_all_clauses if clause[0] > 0]
        else:
            tw_filtered_clauses = [clause for clause in tw_all_clauses if clause[0] < 0]

        tw_clauses_subset = random.sample(tw_filtered_clauses, accumulation)
        document_of_features = []
        for tw_clause in tw_clauses_subset:
            related_literals = tw_clause[1]
            for literal in related_literals:
                literal_knowledge_path = Dicrectories.knowledge_pkl_path_by_id(literal)
                literal_all_clauses = Tools.read_pickle_data(literal_knowledge_path)
                if target_value == 1:
                    literal_filtered_clauses = [clause for clause in literal_all_clauses if clause[0] > 0]
                else:
                    literal_filtered_clauses = [clause for clause in literal_all_clauses if clause[0] < 0]
                
                literal_clauses_subset = random.sample(literal_filtered_clauses, accumulation)
                for literal_clause in literal_clauses_subset:
                    literals = literal_clause[1]
                    for literal in literals:
                        document_of_features.append(literal)

        store_to_X(document_of_features, X)
        documents_X.append(X)
        documents_Y.append(target_value)
    return documents_X, documents_Y

def get_knowledge(_LOGGER, args):
    number_of_documents = 25000
    X_train, Y_train = generate_knowledge(number_of_documents, args)
    X_test, Y_test = generate_knowledge(number_of_documents, args)
    return X_train,Y_train,X_test,Y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=10000, type=int)
    parser.add_argument("--T", default=8000, type=int)
    parser.add_argument("--s", default=2.0, type=float)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--clause_drop_p", default=0.75, type=float)
    parser.add_argument("--max-ngram", default=2, type=int)
    parser.add_argument("--features", default=5000, type=int)
    parser.add_argument("--imdb-num-words", default=5000, type=int)
    parser.add_argument("--imdb-index-from", default=2, type=int)
    args = parser.parse_args()

    # X_train, Y_train, X_test, Y_test = get_imdb(_LOGGER, args)
    X_train, Y_train, X_test, Y_test = get_knowledge(_LOGGER, args)

    _LOGGER.info("Selecting Features.... Done!")

    tm = TMClassifier(args.num_clauses, args.T, args.s, platform=args.device, weighted_clauses=args.weighted_clauses,
                      clause_drop_p=args.clause_drop_p)

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    for epoch in range(args.epochs):
        benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
        with benchmark1:
            tm.fit(X_train, Y_train)

        benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
        with benchmark2:
            result = 100 * (tm.predict(X_test) == Y_test).mean()

        _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                     f"Testing Time: {benchmark2.elapsed():.2f}s")

