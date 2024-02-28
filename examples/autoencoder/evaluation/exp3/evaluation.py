import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import numpy as np

class Evaluation:
    def calculate(self, target_similarity, pair_list):
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