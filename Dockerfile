FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY pyproject.toml .
RUN pip install .

COPY . .
RUN touch .project-root

# --- ADD THIS DEBUG LINE ---
# This prints all files in the image to the build logs
RUN echo "--- FILES IN DOCKER IMAGE ---" && ls -R /app && echo "---------------------------"
# ---------------------------

RUN pip install --no-deps .

EXPOSE 5000

CMD ["python", "app/main.py"]