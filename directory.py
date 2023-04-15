from platform import node as _node

_HOSTNAME = _node()
_PROJECT = 'quantize-torch-models'
_HOST_DIR_CONFIG = {
    "kennardnph":
        {
            "PROJECT_DIR": "/mnt/Data/Projects",
            "PROJECT_STORAGE_DIR": "/mnt/Data/ProjectStorage"
        }
}
_HOST_DIRS = _HOST_DIR_CONFIG[_HOSTNAME]
PROJECT_DIR = _HOST_DIRS['PROJECT_DIR']
PROJECT_STORAGE_DIR = _HOST_DIRS['PROJECT_STORAGE_DIR']


if __name__ == '__main__':
    print(_HOSTNAME)
