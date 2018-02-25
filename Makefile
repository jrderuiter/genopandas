
BRANCH = $(shell git symbolic-ref --short HEAD)

docs-autobuild:
	rm -rf docs/_build
	mkdir -p docs/_build
	sphinx-autobuild docs docs/_build

env:
	conda env create --file environment.yaml

gh-pages:
	git checkout gh-pages
	find ./* -not -path '*/\.*' -prune -exec rm -r "{}" \;
	git checkout $(BRANCH) docs Makefile src AUTHORS.rst \
		CONTRIBUTING.rst HISTORY.rst README.rst
	git reset HEAD
	(cd docs && make html)
	mv -fv docs/_build/html/* ./
	rm -rf docs Makefile src AUTHORS.rst CONTRIBUTING.rst \
		HISTORY.rst README.rst
	touch .nojekyll
	git add -A
	git commit -m "Generated gh-pages for `git log $(BRANCH) -1 --pretty=short --abbrev-commit`" \
		&& git push origin gh-pages ; git checkout $(BRANCH)

test:
	pytest tests
