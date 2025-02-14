# Use an official Python image with Jupyter
FROM python:3.10

# Use your base image
FROM ud31/intro_to_hpc_ai_jupyter

# Set the working directory inside the container
WORKDIR /workspace

# Copy your Python and Jupyter Notebook files into the container
COPY Ethical_AI_&_Future_Trends.ipynb Ethical_AI_&_Future_Trends.py /workspace/


# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (if needed)
COPY . /app

# # Copy the requirements.txt file into the container
# COPY requirements.txt .

# # Install dependencies from requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# Install required Python libraries
RUN pip install --upgrade pip
RUN pip install jupyter pandas numpy scikit-learn tensorflow torch matplotlib seaborn

# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook on container startup
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
