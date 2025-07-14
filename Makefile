install:
	pip install -e .

run:
	python gfpeps_app.py

test:
	python -m pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info/