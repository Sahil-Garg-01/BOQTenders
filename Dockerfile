# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --retries 10 --timeout 300 -r requirements.txt

# Copy the entire project
COPY . .

# HF_API_TOKEN should be set at runtime via environment variable
# For local testing: docker run -e HF_API_TOKEN=your_token ...
# For Hugging Face Spaces: Set as a secret in Space settings

# Expose port for Streamlit (HF Spaces default)
EXPOSE 7860

# Run only Streamlit (simpler, more reliable for HF Spaces)
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]