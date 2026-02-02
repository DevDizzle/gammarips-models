FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    google-cloud-bigquery \
    pandas \
    pandas-gbq \
    db-dtypes \
    pyarrow

COPY main.py .

ENTRYPOINT ["python", "main.py"]

