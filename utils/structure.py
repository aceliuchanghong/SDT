# pip install easy-media-utils
from tree_utils.struct_tree_out import print_tree

path = r'../../SDT'
exclude_dirs_set = {'using_files', '__init__.py', 'static'}
print_tree(directory=path, exclude_dirs=exclude_dirs_set)
