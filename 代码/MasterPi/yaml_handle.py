import yaml
import os

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths for the YAML files
lab_file_path = os.path.join(current_dir, 'lab_config.yaml')
Deviation_file_path = os.path.join(current_dir, 'Deviation.yaml')

def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding='utf-8')
    file_data = file.read()
    file.close()
    
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    
    return data

def save_yaml_data(data, yaml_file):
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(data, file)
    file.close()
