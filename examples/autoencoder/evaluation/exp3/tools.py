import codecs
import pickle
import numpy as np
from functools import lru_cache

class Tools:
    @staticmethod
    def get_dataset_pairs(path):
        fread_simlex=codecs.open(path, 'r', 'utf-8')
        pair_list = []
        line_number = 0
        for line in fread_simlex:
            if line_number > 0:
                tokens = line.split(',')
                word_i = tokens[0].lower()
                word_j = tokens[1].lower()
                score = float(tokens[2].replace('\n', ''))
                pair_list.append( ((word_i, word_j), score) )
            line_number += 1
        return pair_list
    
    @staticmethod
    @lru_cache(maxsize=None)
    def read_pickle_data(path):
        with open(path, "rb") as saved:
            return pickle.load(saved)
    
    @staticmethod
    def generate_targets(base_path):
        word1 = Tools.read_pickle_data(base_path + '_word1.pkl')
        word2 = Tools.read_pickle_data(base_path + '_word2.pkl')
        word_total= list(set(word1 + word2))
        vectorizer_X = Tools.read_pickle_data("vectorizer_X.pickle")
        target_words=[]
        for i in word_total:
            if i in vectorizer_X.vocabulary_:
                target_words.append(i)
        output_active = np.empty(len(target_words), dtype=np.uint32)
        for i in range(len(target_words)):
            target_word = target_words[i]
            target_id = vectorizer_X.vocabulary_[target_word]
            output_active[i] = target_id
        output= open(base_path + '_target.pkl', "wb")
        pickle.dump((output_active, target_words), output)
        output.close()
        return output_active, target_words
    
    @staticmethod
    def get_targets(base_path):
        output_active, target_words = Tools.read_pickle_data(base_path + '_target.pkl')
        return output_active, target_words
    
    @staticmethod
    def print_training_time(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")
