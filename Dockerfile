FROM phusion/baseimage
LABEL maintainer="l.pirch@tu-bs.de"

RUN apt update && apt upgrade -y
RUN apt install -y wget unzip python3.5
RUN ln -s /usr/bin/python3.5 /usr/bin/python

# install python pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py && rm get-pip.py

# copy complete repo
COPY . /informaticup

WORKDIR informaticup
RUN mkdir .cache .process logs data

# force docker CMD instruction to use /bin/bash to support sourcing
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
CMD source /informaticup/setup_project.sh && python /informaticup/manage.py runserver 0.0.0.0:80
