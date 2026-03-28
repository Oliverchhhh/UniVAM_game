r ?= logs
k ?= 1

.PHONY: quality style clean

check_dirs := src scripts train.py eval.py download_models.py
exclude_dirs := tests

quality:
	ruff check $(check_dirs) --exclude $(exclude_dirs)
	ruff format --check $(check_dirs) --exclude $(exclude_dirs)

style:
	ruff check $(check_dirs) --fix --exclude $(exclude_dirs)
	ruff format $(check_dirs) --exclude $(exclude_dirs)

clean:
	@bash scripts/envs/clean_cache.sh -r $(r) -k $(k)
