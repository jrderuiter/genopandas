
docs-autobuild:
	rm -rf docs/_build
	mkdir -p docs/_build
	sphinx-autobuild docs docs/_build

test:
	pytest tests
