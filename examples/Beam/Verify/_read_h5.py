"""Read ProblemDefinition.h5 and print its contents."""
import sys; sys.path.insert(0, '../../src')
import h5py
import numpy as np

f = h5py.File('model/ProblemDefinition.h5', 'r')

def print_tree(g, prefix=''):
    for k in g:
        item = g[k]
        if isinstance(item, h5py.Group):
            print(f"{prefix}{k}/")
            for ak, av in item.attrs.items():
                print(f"{prefix}  @{ak} = {av}")
            print_tree(item, prefix + '  ')
        else:
            d = np.array(item)
            print(f"{prefix}{k}: shape={d.shape}, dtype={d.dtype}")
            if d.size <= 50:
                print(f"{prefix}  values = {d}")
            else:
                print(f"{prefix}  min={d.min():.6e}, max={d.max():.6e}")

print_tree(f)
f.close()
