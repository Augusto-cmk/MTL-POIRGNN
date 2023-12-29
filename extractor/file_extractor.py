import json

import pandas as pd

from foundation.configuration.input import Input

class FileExtractor:

    def __init__(self):
        # self.users_steps_csv_filename = Input.get_instance().inputs['users_steps_filename']
        # self.ground_truth_filename = Input.get_instance().inputs['ground_truth']
        pass

    def read_csv(self, filename):

        df = pd.read_csv(filename)

        return df

    def extract_ground_truth_from_csv(self, filename):
        df = pd.read_csv(filename)

        return df

    def read_json(self, filename):

        with open(filename) as file:
            data = json.loads(file.read())

        return data