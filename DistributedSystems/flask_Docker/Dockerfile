FROM python:3.10.9
LABEL maintainer="Roshny  George"

WORKDIR /app
COPY  requirements.txt .
RUN pip install -r requirements.txt

COPY  app.py .

EXPOSE 6111
# command to run on container start
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
#CMD [ "python3", "app.py"] # not working for flask application
