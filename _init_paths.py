import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

package_path = osp.join(this_dir, '.', 'control')
add_path(package_path)

package_path = osp.join(this_dir, '.', 'graph')
add_path(package_path)

package_path = osp.join(this_dir, '.', 'utils')
add_path(package_path)

# package_path = osp.join(this_dir, '.', 'graph', 'ops')
# add_path(package_path)

# Add lib to PYTHONPATH
# lib_path = osp.join(this_dir, '..', 'lib')
# add_path(lib_path)

# coco_path = osp.join(this_dir, '..', 'data', 'coco', 'PythonAPI')
# add_path(coco_path)
