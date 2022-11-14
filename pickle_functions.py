import os
import pickle

def pickle_dump(object, filename, path):
    """Saves python objects. \\
       Uses the given object's type for the filename extension.

    Args:
        object: The object to be saved in a file.
        filename (str): The name of the file.
        path (str): The directory of the file.
    """
    filehandler = open(os.path.join(path, filename) + '.' + str(type(object).__name__), 'wb')
    pickle.dump(object, filehandler)


def pickle_load(file, path):
    """Imports a python object from a file.

    Args:
        file (str): Filename (including the filename extension).
        path (str): The directory of the file.

    Returns:
        The object of interest.
    """
    filehandler = open(os.path.join(path, file), 'rb')
    object = pickle.load(filehandler)
    return object