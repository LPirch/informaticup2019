FROM python:3.5-slim
LABEL maintainer="l.pirch@tu-bs.de"

RUN apt update && apt upgrade -y
RUN apt install -y wget unzip

# copy complete repo
COPY . /informaticup

WORKDIR informaticup
RUN mkdir .cache .process logs
CMD /bin/sh setup_project.sh && python manage.py runserver 0.0.0.0:80