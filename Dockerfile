FROM python:3.12-slim-bookworm

# Set the working directory
WORKDIR /app

# Install system libraries required by LightGBM
RUN apt-get update && apt-get install -y libgomp1

COPY pyproject.toml .

# Install the dependencies
# will run the command within /app
RUN pip install .

# Copy your application code
COPY . .

# will create the marker file so 'from_root' knows this is the root folder
RUN touch .project-root

RUN pip install --no-deps .

# Expose the port FastAPI will run on
EXPOSE 5000

# the default command to execute when the container starts
CMD ["python", "app/main.py"]