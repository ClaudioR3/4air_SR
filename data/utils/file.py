import os
import shutil
import numpy as np

class FileStorage: 
    """
    This class allows the comunication with the FileSystem.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
    def read(self, file, decode = "utf-8"):
        """
        if decode is None: 
            with open(file, 'rb') as f:
                reader = f.read()
        else:
            with open(file, 'rb') as f:
                reader = f.read().decode(decode)
        return reader
        """
        return np.load(file)
    
    def write(self, file, body):
        """Write the content body in the file using binary mode
        :param file: File path
        """
        try:
            with open(file, 'wb') as f:
                f.write(body)
        except TypeError:
            with open(file, 'wb') as f:
                f.write(body.encode("utf-8"))
            
    def objectslist(self, path):
        """Return the list of files in the directory path
        :param path: the directory path
        :return: the absolute path of each file in the directory path"""
        return [os.path.join(path,file) for file in os.listdir(path)]
    
    def clean(self,path):
        """Clean the directory path
        :param path: the directory path
        :return: None
        """
        if os.path.exists(path):
            shutil.rmtree(path)
    
    def mkdir(self, path):
        """ Create the directory path if not exists
        :param path: The directory path
        :return: None
        """
        if not os.path.exists(path): os.mkdir(path)