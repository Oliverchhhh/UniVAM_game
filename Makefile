r ?= logs
k ?= 1

.PHONY: quality style clean

check_dirs := src train.py
exclude_dirs := tests

quality:
	ruff check $(check_dirs) --exclude $(exclude_dirs)
	ruff format --check $(check_dirs) --exclude $(exclude_dirs)

style:
	ruff check $(check_dirs) --fix --exclude $(exclude_dirs)
	ruff format $(check_dirs) --exclude $(exclude_dirs)

clean:
	@bash scripts/clean_cache.sh -r $(r) -k $(k)
