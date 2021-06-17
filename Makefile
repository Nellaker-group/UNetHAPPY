environment:
	conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	conda install -y -c pyviz holoviews bokeh
	conda install -y datashader
	python setup.py develop
.PHONY: install


environment_cpu:
	conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	conda install -y -c pyviz holoviews bokeh
	conda install -y datashader
	python setup.py develop
.PHONY: install

setup:
	python setup.py develop
.PHONY: install

test:
	pytest
.PHONY: install

fmt:
	black happy projects/placenta analysis
.PHONY: install

