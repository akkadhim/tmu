import codecs
import pickle
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import numpy as np

class Evaluation_Tools:
    def evaluate(self, target_similarity, pair_list):
        calculated_score=[]
        extracted_list = []
        original_score=[]
        word_pairs=[]
        
        for (x,y) in pair_list:
            if x in target_similarity:
                word1, word2=x
                word1_prof = target_similarity[x] 
                extracted_list.append((x, word1_prof))
                calculated_score.append(word1_prof)
                original_score.append(y)
                word_pairs.append(x)

        spearman_TM = spearmanr(original_score, calculated_score)
        spearman_TM = round(spearman_TM[0], 3)
        print(f'Spearman TM: {spearman_TM}')

        total_list=[]
        total_list.append(original_score)
        total_list.append(calculated_score)

        similarity = cosine_similarity(total_list)
        print(f'Cosine TM \n{similarity}')

        TM_corr= np.corrcoef(original_score, calculated_score)
        print(f'Pearson TM \n{TM_corr}')

        kendal_TM, _ = kendalltau(original_score, calculated_score)
        print(f'Kendal TM: {kendal_TM}')

        data = pd.DataFrame([original_score,calculated_score])
        data=data.transpose()
        data.columns=['Original','TM']
        correlation = data.corr()
        print("Pearson Corr \n", correlation)
        
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
        word1 = Evaluation_Tools.read_pickle_data(base_path + '_word1.pkl')
        word2 = Evaluation_Tools.read_pickle_data(base_path + '_word2.pkl')
        word_total= list(set(word1 + word2))
        vectorizer_X = Evaluation_Tools.read_pickle_data("vectorizer_X.pickle")
        target_words=[]
        for i in word_total:
            if i in vectorizer_X.vocabulary_:
                target_words.append(i)
        output= open(base_path + '_target.pkl', "wb")
        pickle.dump(target_words, output)
        output.close()
        return target_words

    