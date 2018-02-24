
docs-autobuild:
	rm -rf docs/_build
	mkdir -p docs/_build
	sphinx-autobuild docs docs/_build

env:
	conda env create --file environment.yaml

test:
	pytest tests
