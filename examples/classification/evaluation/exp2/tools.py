import re
import os
import nltk
import string
import pickle
import numpy as np
from directories import Dicrectories

nltk.download('stopwords')
nltk.download('wordnet')

class Tools:
    @staticmethod
    def split_labels_and_sentences(path):
        f = open(path, "r", encoding="utf-8")
        lines = f.readlines()
        f.close()
        doc_name_list = []
        doc_content_list = []
        for line in lines:
            line = line.strip()
            label = line[:line.find('\t')]
            content = line[line.find('\t') + 1:]
            #string = str(doc_id) + '\t' + 'train' + '\t' + label
            doc_name_list.append(label)
            doc_content_list.append(content)
        return (doc_name_list, doc_content_list)
    
    @staticmethod
    def preprocess(words):
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        # we'll make use of python's translate function,that maps one set of characters to another
        # we create an empty mapping table, the third argument allows us to list all of the characters
        # to remove during the translation process

        # first we will try to filter out some  unnecessary data like tabs
        table = str.maketrans('', '', '\t')
        
        words = [word.translate(table) for word in words]
        words= [lemmatizer.lemmatize(word) for word in words]

        punctuations = (string.punctuation).replace("'", "")
        # the character: ' appears in a lot of stopwords and changes meaning of words if removed
        # hence it is removed from the list of symbols that are to be discarded from the documents
        trans_table = str.maketrans('', '', punctuations)
        stripped_words = [word.translate(trans_table) for word in words]

        # some white spaces may be added to the list of words, due to the translate function & nature of our documents
        # we remove them belowr
        words = [str for str in stripped_words if str]

        # some words are quoted in the documents & as we have not removed ' to maintain the integrity of some stopwords
        # we try to unquote such words below
        p_words = []
        for word in words:
            if (word[0] and word[len(word) - 1] == "'"):
                word = word[1:len(word) - 1]
            elif (word[0] == "'"):
                word = word[1:len(word)]
            else:
                word = word
            p_words.append(word)

        words = p_words.copy()

        # we will also remove just-numeric strings as they do not have any significant meaning in text classification
        words = [word for word in words if not word.isdigit()]

        # we will also remove single character strings
        words = [word for word in words if not len(word) == 1]

        # after removal of so many characters it may happen that some strings have become blank, we remove those
        words = [str for str in words if str]

        # we also normalize the cases of our words
        words = [word.lower() for word in words]

        # we try to remove words with only 2 characters
        words = [word for word in words if len(word) > 2]

        return words

    @staticmethod
    def flatten(list):
        new_list = []
        for i in list:
            for j in i:
                new_list.append(j)
        return new_list

    @staticmethod
    def remove_stopwords(words):
        from nltk.corpus import stopwords
        stopwords = set(stopwords.words('english'))
        words = [word for word in words if not word in stopwords]
        return words

    @staticmethod
    def tokenize_line(line):
        text = line[0:len(line)-1].strip().split(" ")
        text= Tools.remove_stopwords(text)
        text= Tools.preprocess(text)
        return text

    @staticmethod
    def tokenize(text):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        text = Tools.tokenize_line(string)
        return text
    
    @staticmethod
    def encoding_sent(text, word_idx_saved):
        feature_set = np.zeros((len(text), len(word_idx_saved)), dtype=np.uint32)
        tnum=0
        for t in text:
            for w in t:
                if (w in word_idx_saved):
                    idx = word_idx_saved[w]
                    feature_set[tnum][idx-1] = 1
            tnum += 1
        return feature_set
    
    @staticmethod
    def generate_words_map(words):
        if not(os.path.exists('./r8.pkl')):
            word_set=set(words)
            word_idx = dict((c, i + 1) for i, c in enumerate(word_set))
            output= open("./r8.pkl", "wb")
            pickle.dump(word_idx, output)
            output.close()
        else:
            saved= open("./r8.pkl", "rb")
            word_idx= pickle.load(saved)
            saved.close()

        print("word_idx",len(word_idx))
        reverse_word_map = dict(map(reversed, word_idx.items()))
        return word_idx, reverse_word_map
    
    @staticmethod
    def read_pickle_data(path):
        saved = open(path, "rb")
        data = pickle.load(saved)
        saved.close()
        return data
    
    @staticmethod
    def get_related(word, vectorizer_X,top_weight):
        if word in vectorizer_X.vocabulary_:
            knowledge_id = vectorizer_X.vocabulary_[word]
            if Dicrectories.check_pkl_exist(knowledge_id):
                knowledge_filepath = Dicrectories.knowledge_pkl_path_by_id(knowledge_id)
                with open(knowledge_filepath, 'rb') as f:
                    word_clauses = pickle.load(f)
                positive_clauses = [clause for clause in word_clauses if clause[0] > 0]
                sorted_positive_clauses = sorted(positive_clauses, key=lambda x: x[0], reverse=True)[:top_weight]
                all_literals = [literal for tw_clause in sorted_positive_clauses for literal in tw_clause[1]]
                feature_names = vectorizer_X.get_feature_names_out()
                words_for_literals = [feature_names[id] for id in all_literals]
                return words_for_literals

                        