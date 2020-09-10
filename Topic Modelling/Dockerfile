FROM python:3.7-stretch
MAINTAINER Tanmay Thakur <tanmaythakurbrn2rule@gmail.com>

# Install Build Utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

# Check Python Environment
RUN python --version
RUN pip --version

# set the working directory for containers
WORKDIR  /usr/src

# Installing Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying Files
COPY . .

# Running Python Application
CMD ["uvicorn","lda2vec_API:app","--port","8082"] 
