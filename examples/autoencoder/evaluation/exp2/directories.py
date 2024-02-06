import os
import datetime

class Dicrectories:
    datasets = 'datasets'
    user_home = os.path.expanduser("~")

    @staticmethod
    def test(dataset_name,id):
        current_folder_path = os.path.join(Dicrectories.datasets, dataset_name)
        current_time = datetime.datetime.now()
        test_id = current_time.strftime("%Y%m%d%H%M%S")
        result_filename = f"{dataset_name}_{id}_{test_id}"
        test_start_name = os.path.join(current_folder_path, "tests")
        result_filepath = os.path.join(test_start_name , result_filename + '.txt')
        return result_filepath
    
    def knowledge(dataset_name):
        # dataset_path = os.path.join(Dicrectories.user_home, dataset_name)
        dataset_path = os.path.join(Dicrectories.datasets, dataset_name)
        path = os.path.join(dataset_path, "knowledge")
        return  path


