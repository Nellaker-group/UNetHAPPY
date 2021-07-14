environment:
	conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install datashader==0.13.0
	pip install "holoviews[recommended]"
	python setup.py develop
.PHONY: install


environment_cpu:
	conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install datashader==0.13.0
	pip install "holoviews[recommended]"
	python setup.py develop
.PHONY: install

graph_environment:
	pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html
.PHONY: install

graph_environment_cpu:
	pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.8.0+cpu.html
.PHONY: install

setup:
	python setup.py develop
.PHONY: install

test:
	pytest
.PHONY: install

fmt:
	black happy projects/placenta analysis qupath
.PHONY: install

