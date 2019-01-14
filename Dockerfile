FROM python:3.5-slim
LABEL maintainer="l.pirch@tu-bs.de"

RUN apt update && apt upgrade -y
RUN apt install -y wget unzip

# copy complete repo
COPY . /informaticup

WORKDIR informaticup
RUN mkdir .cache .process logs data

# force docker CMD instruction to use /bin/bash to support sourcing
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
CMD set -e && source /informaticup/setup_project.sh && python /informaticup/manage.py runserver 0.0.0.0:80
