import codecs
import pickle

class evaluation_tool:

    @staticmethod
    def get_dataset_pairs(path):
        fread_simlex=codecs.open(path, 'r', 'utf-8')
        pair_list = []
        line_number = 0
        for line in fread_simlex:
            if line_number > 0:
                tokens = line.split(',')
                word_i = tokens[1].lower()
                word_j = tokens[2].lower()
                score = float(tokens[3].replace('\n', ''))
                pair_list.append( ((word_i, word_j), score) )
            line_number += 1
        return pair_list
    
    @staticmethod
    def read_pickle_data(path):
        saved = open(path, "rb")
        data = pickle.load(saved)
        saved.close()
        return data
    
    @staticmethod
    def generate_target_words(base_path):
        word1 = evaluation_tool.read_pickle_data(base_path + '_word1.pkl')
        word2 = evaluation_tool.read_pickle_data(base_path + '_word2.pkl')
        word_total= list(set(word1 + word2))
        vectorizer_X = evaluation_tool.read_pickle_data("vectorizer_X.pickle")
        target_words=[]
        for i in word_total:
            if i in vectorizer_X.vocabulary_:
                target_words.append(i)
        output= open(base_path + '_target.pkl', "wb")
        pickle.dump(target_words, output)
        output.close()
        return target_words