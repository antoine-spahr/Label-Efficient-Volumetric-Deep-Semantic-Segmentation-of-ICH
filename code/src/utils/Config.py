import json

class Config:
    """
    Base class for experimental configuration management.
    """
    def __init__(self, settings=None):
        """
        Build a Config instance.
        """
        self.settings = settings

    def load_config(self, import_path):
        """
        Load the config json from the import path.
        """
        with open(import_path, 'r') as fb:
            self.settings = json.load(fb)

    def save_config(self, export_path):
        """
        Save the config json at the export path.
        """
        with open(export_path, 'w') as fb:
            json.dump(self.settings, fb)
