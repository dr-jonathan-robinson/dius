FROM python:3.11.5-slim

WORKDIR /app


COPY requirements.docker.txt .


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.docker.txt


COPY categorical_mapping.joblib .
COPY dius_model.keras .
COPY serve.py .

EXPOSE 5000

ENV PORT=5000

# Run the application
CMD ["python", "serve.py"]
