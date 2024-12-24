FROM python:3.10-slim

WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir gradio

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# RUN apt-get update && apt-get install -y
CMD ["python", "src/main.py"]
