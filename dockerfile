# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# Copy one explicitly packaged MLflow run with its matching preprocessing schema.
# Create this directory with scripts/package_model.py before building the image.
COPY src/serving/model/production /app/model

# make "serving" and "app" importable without the "src." prefix
# ensures logs are shown in real-time (no buffering).
# lets you import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/model

# 6. Expose FastAPI port
EXPOSE 8000 10000

# Run on Render's PORT, or 8000 for local Docker runs.
CMD ["sh", "-c", "python -m uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
