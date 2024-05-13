FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY app/requirements.txt /src
COPY data/* /src/data/
RUN pip install -r requirements.txt
COPY . .
EXPOSE 80
CMD ["flask", "--app", "app", "run", "--host", "0.0.0.0", "--port", "80"]

