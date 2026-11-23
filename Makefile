# Convenience targets. Every one is a thin wrapper around a script in scripts/,
# so read those for the full option set.

WORK ?= work
DATA ?= data
PYTHON ?= python

.PHONY: help install install-all test lint clean \
        base-dataset cutouts synthetic datasets train-base train-solution \
        evaluate ablations baselines-owlvit baselines-protonet baselines \
        pipeline strip-notebooks

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:            ## Install the package and core dependencies
	pip install -e .

install-all:        ## Install with every optional extra
	pip install -e ".[all]"

test:               ## Run the test suite
	pytest -q

lint:               ## Lint with ruff
	ruff check src scripts tests

base-dataset:       ## Convert Yellow Sticky Traps from VOC to YOLO
	$(PYTHON) scripts/prepare_base_dataset.py \
	  --source $(DATA)/yellow-sticky-traps-dataset --work-dir $(WORK)

cutouts:            ## Extract alpha-matted insect cutouts
	$(PYTHON) scripts/extract_cutouts.py \
	  --aspilanta $(DATA)/aspilanta_raw \
	  --distractors $(DATA)/distractor_raw --work-dir $(WORK)

synthetic:          ## Generate the synthetic trap dataset
	$(PYTHON) scripts/generate_synthetic.py \
	  --backgrounds $(DATA)/backgrounds --work-dir $(WORK)

datasets:           ## Build the four experiment datasets
	$(PYTHON) scripts/build_datasets.py \
	  --real-aspilanta $(DATA)/real_aspilanta --work-dir $(WORK)

train-base:         ## Stage 1: three-class base detector
	$(PYTHON) scripts/train.py --stage base --work-dir $(WORK)

train-solution:     ## Stage 2: four-class proposed detector
	$(PYTHON) scripts/train.py --stage solution --work-dir $(WORK)

evaluate:           ## Evaluate the trained solution
	$(PYTHON) scripts/evaluate.py --work-dir $(WORK) \
	  --weights $(WORK)/runs/detect/solution_4class_detector/weights/best.pt

ablations:          ## Run the nine ablations (long; resumable)
	$(PYTHON) scripts/run_ablations.py --work-dir $(WORK)

baselines-owlvit:   ## OWL-ViT foundation-model baseline
	$(PYTHON) scripts/run_baselines.py --baseline owlvit --work-dir $(WORK)

baselines-protonet: ## Prototypical-network baseline vs frozen CLIP
	$(PYTHON) scripts/run_baselines.py --baseline protonet --work-dir $(WORK)

baselines: baselines-owlvit baselines-protonet  ## Both standalone baselines
	$(PYTHON) scripts/train.py --stage tfa --work-dir $(WORK)

pipeline: base-dataset cutouts synthetic datasets train-base train-solution evaluate  ## Full pipeline end to end

strip-notebooks:    ## Strip outputs from the archived notebooks
	nbstripout notebooks/archive/*.ipynb

clean:              ## Remove caches and build products
	rm -rf build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
