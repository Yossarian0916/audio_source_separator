import os


def get_root_path():
    current_file_path = os.path.abspath(__file__)
    root = os.path.dirname(os.path.dirname(current_file_path))
    return root


def get_data_path():
    root = get_root_path()
    return os.path.join(root, 'data')


def get_saved_model_path():
    root = get_root_path()
    return os.path.join(root, 'saved_model')


def get_models_path():
    root = get_root_path()
    return os.path.join(root, 'models')


def get_training_path():
    root = get_root_path()
    return os.path.join(root, 'training')


def get_evaluation_path():
    root = get_root_path()
    return os.path.join(root, 'evaluation')
