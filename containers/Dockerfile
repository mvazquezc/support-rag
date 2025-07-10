FROM quay.io/fedora/python-313:latest
MAINTAINER Mario Vazquez <mavazque@redhat.com>

USER root
RUN mkdir -p /opt/ragapi

COPY ../requirements.txt /opt/ragapi/requirements.txt

WORKDIR /opt/ragapi

RUN pip3 install -r /opt/ragapi/requirements.txt

USER 1024
ADD ../src/ /opt/ragapi/

ENTRYPOINT ["python3", "-u", "/opt/ragapi/cli.py"]