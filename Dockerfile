FROM python:3.9.7

WORKDIR /

RUN pip3 install flask

COPY . .

EXPOSE 5005
CMD [ "python", "app.py"]