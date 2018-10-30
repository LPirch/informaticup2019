FROM django
LABEL maintainer="l.pirch@tu-bs.de"

# copy complete repo
COPY . /informaticup
RUN pip install --upgrade pip
RUN pip install -r informaticup/pip_requirements.txt