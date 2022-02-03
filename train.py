from __future__ import print_function

import os
import json
import pickle
import sys
import traceback

import pandas as pd
import shutil
from sklearn import tree
import argparse

import yaml
import collections

import glob

import time


# These are the paths to where SageMaker mounts interesting things in your container.
prefix = '/opt/ml/'
input_path = prefix + 'input/data'

scripts_path = os.path.join(input_path, 'scripts')

output_path = os.path.join(prefix, 'output')

model_path="/workspace/foxp2/checkpoints"
#syncnet_checkpoint_path=os.path.join(input_path, 'syncnet_checkpoint/syncnet_checkpoint.pth')
#disc_checkpoint_path = os.path.join(input_path, 'pretrained/disc_checkpoint.pth')
#generator_checkpoint_path = os.path.join(input_path, 'pretrained/generator_checkpoint.pth')

# The function to execute the training.
def train(script_name,restore_path):
    print('Starting the training.')
    try:
        print("Starting to wait")
        time.sleep(60)
        print("1 minute wait ended")
        shutil.copyfile(os.path.join(scripts_path,script_name), "/workspace/TTS/train_script.py")
        print("{} copied".format(script_name))
        if restore_path!="no":
            shutil.copyfile(restore_path, "/workspace/TTS/copied_checkpoint.pth.tar")
            print("{} copied".format(restore_path))
        os.environ["CUDA_VISIBLE_DEVICES"]="0, 1 , 2, 3"
        print("Cuda visible devices put to 4")
        if restore_path!="no":
            os.system("python3 distribute-Copy.py --script train_script.py --restore_path copied_checkpoint.pth.tar")
        else:
            os.system("python3 distribute-Copy.py --script train_script.py")
        print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
        
#as per https://github.com/aws/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/keras_bring_your_own/trainer/environment.py
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_hyperparameters():
    return HyperParameters(load_config(os.path.join('/opt/ml/input/config', 'hyperparameters.json')))
    
class HyperParameters(collections.Mapping):
    """dict of the hyperparameters provided in the training job. Allows casting of the hyperparameters
    in the `get` method.
    """

    def __init__(self, hyperparameters_dict):
        self.hyperparameters_dict = hyperparameters_dict

    def __getitem__(self, key):
        return self.hyperparameters_dict[key]

    def __len__(self):
        return len(self.hyperparameters_dict)

    def __iter__(self):
        return iter(self.hyperparameters_dict)

    def get(self, key, default=None, object_type=None):
        """Has the same functionality of `dict.get`. Allows casting of the values using the additional attribute
        `object_type`:
        Args:
            key: hyperparameter name
            default: default hyperparameter value
            object_type: type that the hyperparameter wil be casted to.
        Returns:
        """
        try:
            value = self.hyperparameters_dict[key]
            if not object_type:
                return value
            elif object_type == bool:
                if value.lower() in ["True", "true"]:
                    return True
                return False
            else:
                return object_type(value)
        except KeyError:
            return default

    def __str__(self):
        return str(self.hyperparameters_dict)

    def __repr__(self):
        return str(self.hyperparameters_dict)




if __name__ == '__main__':
    hyperparameters=load_hyperparameters()
    train(
        hyperparameters['script_name'],
        hyperparameters['restore_path'])

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)