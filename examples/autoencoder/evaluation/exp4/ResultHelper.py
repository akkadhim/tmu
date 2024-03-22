import os
import re
from scipy.stats import spearmanr

class ResultHelper:
    @staticmethod
    def get_file_max_spearman(file_path):
        max_spearman = 0
        with open(file_path, 'r') as f:
            content = f.read()
            matches = re.findall(r'Spearman TM: ([\d.]+)', content)
            for match in matches:
                spearman_val = float(match)
                if spearman_val > max_spearman:
                    max_spearman = spearman_val
        return max_spearman
    
    @staticmethod
    def get_dataset_max_spearman(folder_path, text_has=None, text_not=None):
        max_dataset_spearman = 0
        file_name = ""
        for file in os.listdir(folder_path):
            if file.endswith(".txt"):
                if (text_has is None or text_has in file.lower()) and (text_not is None or text_not not in file.lower()):
                    file_path = os.path.join(folder_path, file)
                    max_spearman = ResultHelper.get_file_max_spearman(file_path)
                    if(max_spearman > max_dataset_spearman):
                        max_dataset_spearman = max_spearman
                        file_name = file
        return max_dataset_spearman, file_name

    @staticmethod
    def search_result(text_has=None, text_not=None):
        datasets_path = "datasets"
        for dataset_folder in os.listdir(datasets_path):
            dataset_path = os.path.join(datasets_path, dataset_folder)
            if os.path.isdir(dataset_path):
                test_path = os.path.join(dataset_path, "tests")
                if os.path.isdir(test_path):
                    max_spearman, file_name = ResultHelper.get_dataset_max_spearman(test_path, text_has = text_has, text_not = text_not)
                    print("Maximum Spearman result for dataset {:15s}: {:3.2f} (File: {})".format(dataset_folder, max_spearman, file_name))