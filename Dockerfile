# Dockerfile
FROM python:3.11-slim

# Set envs for logging and performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 80

# Run with Gunicorn + Uvicorn worker (90s timeout)
CMD ["gunicorn", "--bind", "0.0.0.0:80", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "90", "-w", "2", "main:app"]
