PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup data run quick test app notebook clean

setup:            ## create venv + install deps
	python3 -m venv .venv && $(PIP) install -q --upgrade pip && $(PIP) install -q -r requirements.txt

data:             ## download the ULB dataset from OpenML into data/creditcard.csv
	$(PY) -c "import sys; sys.path.insert(0,'src'); from fraudpipe.data import load_creditcard; df=load_creditcard('data'); print(df.shape)"

run:              ## full pipeline -> results/
	$(PY) run.py

quick:            ## 20 % sample, no SHAP (about a minute)
	$(PY) run.py --sample-frac 0.2 --skip-shap --out results_quick

test:
	$(PY) -m pytest -q

app:              ## analyst review dashboard
	.venv/bin/streamlit run app/streamlit_app.py

notebook:         ## build + execute the EDA notebook
	$(PY) notebooks/build_eda.py && .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

clean:
	rm -rf results_quick .pytest_cache
