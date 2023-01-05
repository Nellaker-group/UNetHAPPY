environment_cu111:
	pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install datashader==0.13.0
	pip install "holoviews[recommended]"
	pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
	pip install -e .
.PHONY: install

environment_cpu:
	conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install datashader==0.13.0
	pip install "holoviews[recommended]"
	pip install -e .
.PHONY: install

setup:
	pip install -e .
.PHONY: install

test:
	pytest
.PHONY: install

fmt:
	black happy projects/placenta analysis qupath
.PHONY: install

