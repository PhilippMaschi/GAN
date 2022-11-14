import os

class Paths:
    def __init__(self):
        self.cwd = os.getcwd()
        self.data_path = self.create_folder(os.path.join(self.cwd, 'data'))
        self.output_path = self.create_folder(os.path.join(self.cwd, 'output'))
        self.ENERCOOP_output_path = self.create_folder(os.path.join(self.output_path, 'ENERCOOP'))
    
    def create_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path