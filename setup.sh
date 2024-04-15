#!/bin/bash

VENV=$1

if [ "$VENV" == "--venv" ]; then
    virtualenv -p python3 .
    source ./bin/activate
elif [ "$VENV" == "--conda" ]; then
    conda create -n ag python=3.10.12
    source ~/miniconda3/bin/activate ag
elif [ "$VENV" == "--docker" ]; then
    git clone --single-branch --branch v1.1.0 https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/geosolver.git
    docker build --no-cache . -t registry-cbu.huawei.com/ukrc-k8s/alphageometry_pt:latest
    docker run --name alphageo --gpus="all" -ti --rm --mount type=bind,src=.,target=/ag/ --entrypoint python registry-cbu.huawei.com/ukrc-k8s/alphageometry_pt:latest \
        common_folder_downloader.py --region cn-southwest-2 --app_token 82aaeb97-6bbb-4a9a-a164-07268a0a6d0b --bucket_name bucket-pangu-green-guiyang --path philipjg/pt_ckpt/
    rm -rf ./geosolver/
    mkdir results
    exit
else
    echo "ERROR: needs to be run with --venv , --conda , or --docker"
    exit
fi

pip install -e .[download,torch,geosolver]

python common_folder_downloader.py --region cn-southwest-2 --app_token 82aaeb97-6bbb-4a9a-a164-07268a0a6d0b --bucket_name bucket-pangu-green-guiyang --path philipjg/pt_ckpt/
DATA=pt_ckpt
if [ ! -d "$DATA" ]; then
  cp -r /nfs/ainlp/math_agent/pt_ckpt .
fi
