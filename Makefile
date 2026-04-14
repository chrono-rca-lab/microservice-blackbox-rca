.PHONY: setup deploy inject-fault run-rca evaluate clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

deploy:
	bash infra/deploy-boutique.sh
	bash infra/deploy-monitoring.sh

PYTHON := $(abspath .venv/bin/python3)

inject-fault:
	$(PYTHON) fault_injection/inject.py $(ARGS)

run-rca:
	$(PYTHON) -m rca_engine $(ARGS)

evaluate:
	$(PYTHON) eval/run_experiment.py $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf experiments/run_*
