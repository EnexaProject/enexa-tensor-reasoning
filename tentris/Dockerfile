FROM ubuntu:22.04


RUN apt update
RUN apt install python3.10 -y
RUN apt install curl -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN pip install numpy

COPY ./version1/ ./home/

COPY ./tentris-8.0.0-ubuntu-22.04.zip ./home/
COPY ./tentris/test.py ./home/

RUN pip install ./home/tentris-8.0.0-ubuntu-22.04.zip

RUN pip install networkx
RUN export PYTHONPATH=$PYTHONPATH:/home

ENTRYPOINT bash

# ENTRYPOINT python3.10
# export PYTHONPATH=$PYTHONPATH:/path/to/tnreason