FROM nvidia/cuda:9.0-base
LABEL maintainer="l.pirch@tu-bs.de"
LABEL application="informaticup"

RUN apt update
RUN apt install -y python3.5 python3-pip

# install pip and virtualenv
RUN python3.5 -m pip install --upgrade pip
RUN python3.5 -m pip install --user virtualenv
ENV PATH="/root/.local/bin:${PATH}"

# copy complete repo
COPY . /informaticup
RUN pip install -r informaticup/pip_requirements.txt