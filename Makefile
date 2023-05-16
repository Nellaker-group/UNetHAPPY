environment_cu117:
	pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
	pip install torch_geometric==2.3.1
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
	pip install -r requirements.txt
	pip install pyvips==2.2.1
	pip install -e .
.PHONY: install

environment_cpu:
	pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
	pip install torch_geometric==2.3.1
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
	pip install -r requirements.txt
	pip install pyvips==2.1.14
	pip install -e .
.PHONY: install

environment_cu117_torch2_py10:
	pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
	pip install torch_geometric==2.3.1
	pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
	pip install -r requirements_py10.txt
	pip install pyvips==2.2.1
	pip install datashader==0.14.4
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

