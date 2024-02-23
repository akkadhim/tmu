import os
import datetime

class Dicrectories:
    datasets = 'datasets'
    user_home = os.path.expanduser("~")
    knowledge = os.path.join(user_home, "knowledge")

    @staticmethod
    def test(dataset_name,id):
        current_folder_path = os.path.join(Dicrectories.datasets, dataset_name)
        current_time = datetime.datetime.now()
        test_id = current_time.strftime("%Y%m%d%H%M%S")
        result_filename = f"{dataset_name}_{id}_{test_id}"
        test_start_name = os.path.join(current_folder_path, "tests")
        result_filepath = os.path.join(test_start_name , result_filename + '.txt')
        return result_filepath
    
    @staticmethod
    def knowledge_pkl_path_by_id(id, path = None):
        if path == None:
            path = Dicrectories.knowledge
        return os.path.join(path , str(id) + '.pkl')

    @staticmethod
    def check_pkl_exist(id):
        path = os.path.join(Dicrectories.knowledge , str(id) + '.pkl')
        if os.path.exists(path):
            return True
        else:
            return False