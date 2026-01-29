# Use Python 3.11-slim as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Adding gradio explicitly since it was missing from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gradio

# Copy the rest of the application code
COPY . .

# Expose Gradio port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
