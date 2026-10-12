# AMR-OHEWS -- One Health early warning for antimicrobial resistance
PY      := .venv/bin/python
PIP     := .venv/bin/pip
EXPORT  := PYTHONPATH=src
TAG     ?= full

.PHONY: help setup setup-gpu-volta data quick full real glass sweep real-eval kaggle figures dashboard docs clean

help:
	@echo "make setup      create .venv and install pinned dependencies"
	@echo "make setup-gpu-volta  as above, with the cu126 torch build for V100s"
	@echo "make data       generate one simulated world + sanity figure"
	@echo "make quick      3 seeds, no deep models (~7 min sanity run)"
	@echo "make full       30 seeds, all four tiers + ablations"
	@echo "make real       download NCBI (3 organisms) and run the premise test"
	@echo "make glass      WHO GLASS driver analysis (consumption vs resistance)"
	@echo "make sweep      spatial-transmission sweep for the graph tier"
	@echo "make real-eval  early-warning evaluation on real NCBI data"
	@echo "make kaggle     reproducible Kaggle dataset survey"
	@echo "make figures    render report figures from results/"
	@echo "make dashboard  launch the Streamlit dashboard"
	@echo "make docs       build docs/proposal.docx and docs/proposal.pdf"

setup:
	@# The system python3 lacks ensurepip on this host, so the venv is created
	@# from the conda interpreter, which has it.
	@test -d .venv || $$(command -v python3.13 || echo $$HOME/miniconda3/bin/python) -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt
	@$(EXPORT) $(PY) -c "import sys;sys.path.insert(0,'src');import torch;\
	from amrews.models.temporal import default_device;\
	print('torch',torch.__version__,'-> device',default_device())"

setup-gpu-volta: setup
	@# Volta cards (V100, sm_70) need the cu126 build; see requirements.txt.
	$(PIP) install -q --index-url https://download.pytorch.org/whl/cu126 \
		--force-reinstall torch==2.14.0
	@$(EXPORT) $(PY) -c "import sys;sys.path.insert(0,'src');\
	from amrews.models.temporal import default_device;print('device:',default_device())"

data:
	$(EXPORT) $(PY) -m amrews.data.simulate --seeds 1 --plot

quick:
	$(EXPORT) $(PY) -W ignore -m amrews.eval.experiments --quick --tag quick

full:
	$(EXPORT) $(PY) -W ignore -m amrews.eval.experiments --full --tag full

real:
	$(EXPORT) $(PY) -W ignore -m amrews.data.ncbi --organism Campylobacter Salmonella Klebsiella

glass:
	$(EXPORT) $(PY) -W ignore -m amrews.data.glass

sweep:
	$(EXPORT) $(PY) -W ignore -u -m amrews.eval.experiments --sweep --seeds 12 --tag sweep

real-eval:
	$(EXPORT) $(PY) -W ignore -m amrews.eval.experiments --real --tag real

kaggle:
	$(EXPORT) $(PY) scripts/kaggle_probe.py

figures:
	$(EXPORT) $(PY) -W ignore -m amrews.plots --tag $(TAG)

dashboard:
	$(EXPORT) .venv/bin/streamlit run src/amrews/dashboard/app.py

docs:
	$(EXPORT) $(PY) scripts/build_docs.py

clean:
	rm -rf results/* figures/* data/processed/*
