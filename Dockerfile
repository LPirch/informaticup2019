FROM python:3.5-slim
LABEL maintainer="l.pirch@tu-bs.de"

RUN apt update 
RUN apt install -y wget unzip

# copy complete repo
COPY . /informaticup
RUN pip install --upgrade pip
RUN pip install -r informaticup/pip_requirements.txt

EXPOSE 8080
EXPOSE 6006
WORKDIR informaticup
RUN mkdir .cache .process logs data
RUN python manage.py migrate
CMD /bin/sh setup_project.sh && python manage.py runserver 0.0.0.0:80