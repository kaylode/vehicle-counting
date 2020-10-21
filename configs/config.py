import yaml

class Config():
    def __init__(self, yaml_path):
        yaml_file = open(yaml_path)
        self._attr = yaml.load(yaml_file, Loader=yaml.FullLoader)['settings']

    def __getattr__(self, attr):
        try:
            return self._attr[attr]
        except KeyError:
            return None
