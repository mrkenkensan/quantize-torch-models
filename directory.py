from platform import node as _node
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
_HOSTNAME = _node()
_PROJECT = 'quantize-torch-models'
_HOST_DIR_CONFIG = {
    "kennardnph":
        {
            "PROJECT_DIR": "/mnt/Data/Projects",
            "PROJECT_STORAGE_DIR": "/mnt/Data/ProjectStorage"
        },
    "vultr":
        {
            "PROJECT_DIR": "/mnt/root/Projects",
            "PROJECT_STORAGE_DIR": "/mnt/root/ProjectStorage"
        }
}
_HOST_DIRS = _HOST_DIR_CONFIG[_HOSTNAME]
PROJECT_DIR = os.path.join(_HOST_DIRS['PROJECT_DIR'], _PROJECT)
PROJECT_STORAGE_DIR = os.path.join(_HOST_DIRS['PROJECT_STORAGE_DIR'], _PROJECT)


if __name__ == '__main__':
    print(_HOSTNAME)
