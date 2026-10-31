.PHONY: install data benchmark test api demo docker
install:
	pip install -e ".[api,demo,dev]"
data:
	isp generate-data
benchmark:
	isp benchmark
test:
	pytest -q
api:
	uvicorn incident_severity.api:app --host 0.0.0.0 --port 8000
demo:
	streamlit run app/streamlit_app.py
docker:
	docker build -t incident-severity-api .
