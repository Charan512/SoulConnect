# Use python base image
FROM python:3.11-slim

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy requirements file first
COPY --chown=user requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user . /app/

# HF Spaces expose applications precisely on port 7860
EXPOSE 7860

# Command to run FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
