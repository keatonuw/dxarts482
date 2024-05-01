FROM python:3.11
WORKDIR /src
COPY app/requirements.txt /src
RUN pip install -r requirements.txt
COPY . .
EXPOSE 80
CMD ["flask", "--app", "app", "run"]
