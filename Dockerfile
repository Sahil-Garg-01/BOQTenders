# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# HF_API_TOKEN should be set at runtime via environment variable
# For local testing: docker run -e HF_API_TOKEN=your_token ...
# For Hugging Face Spaces: Set as a secret in Space settings

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Create a startup script
RUN echo '#!/bin/bash\n\
uvicorn app:app --host 0.0.0.0 --port 8000 & \n\
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
wait' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]