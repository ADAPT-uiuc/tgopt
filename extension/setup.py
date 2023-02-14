import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

comp_args = ['-fopenmp']
link_args = ['-fopenmp', '-ltbb']

supported_env_vars = [
    'tgopt_1t_sampling',        # Single-threaded sampling
    'tgopt_1t_cache_keys',      # Single-threaded for computing cache keys
    'tgopt_1t_cache_store',     # Single-threaded for cache store
    'tgopt_1t_cache_lookup',    # Single-threaded for cache lookup
    'tgopt_embed_store_dev',    # Store embeddings on compute device
    'tgopt_force_single',       # Force every operation to be single-threaded
]

for var in supported_env_vars:
    if os.environ.get(var):
        if var == 'tgopt_force_single':
            comp_args.remove('-fopenmp')
            link_args.remove('-fopenmp')
        else:
            comp_args.append(f'-D{var}')

setup(
    name='tgopt_ext',
    version='0.1.0',
    ext_modules=[
        CppExtension('tgopt_ext', ['tgopt_ext.cpp'],
            extra_compile_args = comp_args,
            extra_link_args = link_args),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
