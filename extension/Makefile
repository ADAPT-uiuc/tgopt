.PHONY: default parallel single cpu gpu clean

default: parallel

parallel:
	python setup.py install

single:
	env tgopt_force_single=1 python setup.py install

# This target is specific to the CPU server environment in our paper.
cpu:
	env tgopt_1t_cache_keys=1 tgopt_1t_cache_store=1 python setup.py install

# This target is specific to the GPU machine environment in our paper.
gpu:
	python setup.py install

clean:
	rm -rf build dist tgopt_ext.egg-info
