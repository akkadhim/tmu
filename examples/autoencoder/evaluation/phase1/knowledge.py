import os
import numpy as np
import pickle
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from tqdm import tqdm
from time import time

class Knowledge:
    def __init__(self, clause_weight_threshold, number_of_examples, accumulation,
                 clause_drop_p, factor, T, s, epochs):
        self.clause_weight_threshold = clause_weight_threshold
        self.number_of_examples = number_of_examples
        self.accumulation = accumulation
        self.clause_drop_p = clause_drop_p
        self.factor = factor
        self.clauses = int(factor * 20 / (1.0 - clause_drop_p))
        self.T = T
        self.s = s
        self.epochs = epochs

    def generate(self, feature_names, X_train, current_folder_path, i):
        target_word_clauses = []
        single_output_active = np.empty(1, dtype=np.uint32)
        single_output_active[0] = i
        tm = TMAutoEncoder(self.clauses, self.T, self.s, single_output_active, max_included_literals=3, accumulation=self.accumulation, feature_negation=False, platform='CPU', output_balancing=0.5)
                            
        training_time = 0
        epochs_progress_bar = tqdm(total=self.epochs, desc="Running Epochs")
        for e in range(self.epochs):
            start_training = time()
            tm.fit(X_train, number_of_examples=self.number_of_examples)
            stop_training = time()
            training_time = training_time + (stop_training-start_training)
            epochs_progress_bar.update(1)
        epochs_progress_bar.close()
        seconds = training_time
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        print(f"Training duration: {hours} hours, {minutes} minutes, {seconds} seconds")

        print("=====================================\nClauses\n=====================================")
        for j in range(self.clauses):
            print("Clause #%-2d " % (j), end=' ')
            weight = 0
            weight = tm.get_weight(0, j)
            print("W:%-5d " % (weight), end=' ')
            l = [] 
            related_literals = []
            number_of_literals = 0 
            for k in range(tm.clause_bank.number_of_literals):
                if tm.get_ta_action(j, k) == 1:
                    number_of_literals = number_of_literals + 1
                    related_literals.append(k)
                    if k < tm.clause_bank.number_of_features:
                        l.append("%s(%d)" % (feature_names[k], tm.clause_bank.get_ta_state(j, k)))
                    else:
                        l.append("¬%s(%d)" % (feature_names[k-tm.clause_bank.number_of_features], tm.clause_bank.get_ta_state(j, k)))
            print(": No of features:%-6d" % (number_of_literals), end=" ==> ")
            try:
                print(" ^ ".join(l))
            except UnicodeEncodeError:
                print(" exception ")
            target_word_clauses.append([weight, related_literals])
                            
        knowledge_start_name = os.path.join(current_folder_path, "knowledge")
        knowledge_filepath = os.path.join(knowledge_start_name , str(i) + '.pkl')
        with open(knowledge_filepath , "wb") as phase1file:
            pickle.dump(target_word_clauses, phase1file)
        return training_time, target_word_clauses