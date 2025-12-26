# Dockerfile
FROM python:3.11-slim

# Don't write .pyc files + flush output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Render assigns a dynamic port
ENV PORT=10000
EXPOSE 10000

# Start server (Gunicorn + Uvicorn worker)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT", "--workers", "2", "--timeout", "90"]
