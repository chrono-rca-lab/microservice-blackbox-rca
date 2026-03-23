.PHONY: setup deploy inject-fault run-rca evaluate clean

setup:
	pip install -r requirements.txt

deploy:
	bash infra/deploy-boutique.sh
	bash infra/deploy-monitoring.sh

inject-fault:
	python fault_injection/inject.py $(ARGS)

run-rca:
	python -m rca_engine $(ARGS)

evaluate:
	python eval/run_experiment.py $(ARGS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf experiments/run_*
