from data.utils.file import FileStorage
import boto3
from botocore.exceptions import ClientError
import warnings
import numpy as np
from io import BytesIO


class CephStorage(FileStorage):
    """
    This class allows the comunication with CephObjectStorage.
    """
    
    def __init__(self, root_dir, ceph_properties = "./ceph.properties"):
        super().__init__(root_dir)
        with open(ceph_properties, 'r') as prop:
            l = [line.split("=") for line in prop.readlines()]
            self.properties = {key.strip(): value.strip() for key,value in l}
        self.s3 = boto3.client('s3',
                               aws_access_key_id = self.properties['access_key'],
                               aws_secret_access_key = self.properties['secret_key'],
                               endpoint_url = self.properties['gateway'])
        
    def read(self, file, decode = "utf-8"):
        """Read a S3 object
        :param key: the object's key id
        :return: S3 Object content
        """
        if decode is None: 
            content = self.s3.get_object(
                Bucket = self.properties['bucket_name'], 
                Key = file
            )['Body'].read()
        else:
            content = self.s3.get_object(
                Bucket = self.properties['bucket_name'],
                Key = file
            )['Body'].read().decode(decode)
        with BytesIO(content) as f:
            f.seek(0)
            array = np.load(f)
        return array
    
    def write(self, file, body, acl = 'public-read'):
        """Load a file to S3 / Ceph Object Storage

        :param body: the file's content
        :param key: the file's key
        :return: None
        """
        self.s3.put_object(
            ACL = acl,
            Body=body, 
            Bucket = self.properties['bucket_name'],
            Key = file
        )
        
    def objectslist(self, path):
        """List of S3 objects
        :param path: the S3 prefix
        :return: S3 Objects list
        """
        objs = boto3.resource('s3',
                             aws_access_key_id = self.properties['access_key'],
                             aws_secret_access_key = self.properties['secret_key'],
                             endpoint_url = self.properties['gateway']
                            ).Bucket(self.properties['bucket_name']).objects.filter(Prefix = path).all()
        
        return [obj.key for obj in objs]
    
    
    
    def clean(self, path):
        """Clean S3 data with prefix path
        :param path: the S3 prefix
        :return: None
        """
        warnings.warn('Cleaning torch data path: {} ...'.format(path))
        boto3.resource('s3',
                       aws_access_key_id = self.properties['access_key'],
                       aws_secret_access_key = self.properties['secret_key'],
                       endpoint_url = self.properties['gateway']
                      ).Bucket(self.properties['bucket_name']).objects.filter(
            Prefix = path
        ).delete()
        warnings.warn('...Done.')
        
    def mkdir(self, path):
        """ Void method
        :param path: the S3 prefix
        :return: None
        """
        pass