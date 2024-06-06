# Use the official Python image as a base
FROM python:3.9-slim

# Set environment variables to prevent Python from writing pyc files to disc and buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    unixodbc \
    unixodbc-dev \
    gcc \
    g++ \
    curl \
    gnupg \
    certbot \
    openssl && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the entire application directory into the container
COPY . /app/

# Obtain SSL certificates using Certbot
RUN certbot certonly --standalone --non-interactive --agree-tos -m josvapstg@gmail.com -d metrosegurocdmx.cloud

# Expose the Streamlit default port
EXPOSE 80
EXPOSE 443

# Run the Streamlit app with SSL
CMD ["streamlit", "run", "app.py", "--server.port=443", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.headless=true", "--server.sslCertFile=/etc/letsencrypt/live/metrosegurocdmx.cloud/fullchain.pem", "--server.sslKeyFile=/etc/letsencrypt/live/metrosegurocdmx.cloud/privkey.pem"]
