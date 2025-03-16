<<<<<<< HEAD
FROM python:3.11-slim

# Working directory
WORKDIR /app
COPY . /app

# Install packages as specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter Lab set on port 8888
EXPOSE 8888

# Running Jupyter Lab 
=======
FROM python:3.11-slim

# Working directory
WORKDIR /app
COPY . /app

# Install packages as specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter Lab set on port 8888
EXPOSE 8888

# Running Jupyter Lab 
>>>>>>> e179f08 (second commit DeepLearningAI)
CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]