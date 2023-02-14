FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y vim curl make g++ libtbb-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sL -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p ~/.conda && \
    rm ~/miniconda.sh && \
    ~/.conda/bin/conda init

RUN ~/.conda/bin/conda run -n base \
    pip install torch==1.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /requirements.txt
RUN ~/.conda/bin/conda run -n base \
    pip install -r /requirements.txt && \
    rm /requirements.txt

ADD extension /extension
RUN cd /extension && \
    ~/.conda/bin/conda run -n base \
    python setup.py install && \
    rm -rf /extension
