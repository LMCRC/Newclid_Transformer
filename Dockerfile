FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /ag
ADD . .

RUN apt-get update
RUN apt-get install -y python3 pip python-is-python3 git
RUN pip install --upgrade pip

WORKDIR /ag/geosolver
RUN pip install -e .

WORKDIR /ag
RUN pip install -e .[torch,download]
RUN rm -rf problems_datasets/ results/

ENTRYPOINT ["alphageo"]
