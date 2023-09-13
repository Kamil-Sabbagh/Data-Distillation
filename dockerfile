# Start from an official Miniconda3 base image
FROM continuumio/miniconda3

# Set working directory in docker
WORKDIR /app

# Copy the environment file to the working directory
COPY environment.yml .

# Install the conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "DD", "/bin/bash", "-c"]

# Copy the content of the local src directory to the working directory
COPY src/ .

# Specify the command to run on container start
CMD ["conda", "run", "-n", "your-environment-name", "python", "./your-script.py"]
