all: init format lint

.PHONY: format
format:
	black . --line-length 104
	isort .
	nbqa black .
	nbqa isort .

.PHONY: lint
lint:
	pytest src/ --pylint --flake8 --ignore=src/meta_rl/envs

.PHONY: init
init:
	pip install -U pip
	pip install -e .
	pip install -r requirements.txt
	python3 ./scripts/download-torch.py
	conda install -y tensorboard
	jupyter contrib nbextension install --user
	jupyter nbextensions_configurator enable --user
	python3 -m ipykernel install --user

.PHONY: init-dev
init-dev:
	make init
	pip install -r requirements-dev.txt
	bash ./scripts/install.sh
