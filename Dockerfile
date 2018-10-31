FROM python:3.5-slim
LABEL maintainer="l.pirch@tu-bs.de"

# copy complete repo
COPY . /informaticup
RUN pip install --upgrade pip
RUN pip install -r informaticup/pip_requirements.txt

EXPOSE 8000
WORKDIR informaticup
CMD python manage.py runserver 0.0.0.0:80