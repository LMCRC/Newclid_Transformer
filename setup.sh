#!/bin/bash


VALID_ARGS=$(getopt -o vcdnh --long venv,conda,docker,no-download,help -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

VENV=""
DO_DOWNLOAD=1

help() {
    echo "setup.sh <mode> [--no-download] [--help]"
    echo "mode has to be one of:"
    echo "    -v / --venv -- install with python virtualenv"
    echo "    -c / --conda -- install with miniconda"
    echo "    -d / --docker -- install with docker"
    echo "--no-download -- do not (re-)download model weights and tokenizer after install"
    echo "--help -- disply this help"
    exit
}

env_error(){
    echo "ERROR: Exactly one build environment has to be set"
    exit
}

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -h | --help)
        help
        shift
        ;;
    -v | --venv)
        if [ ! -z "$VENV" ]; then
            env_error
        fi
        VENV="venv"
        shift
        ;;
    -c | --conda)
        if [ ! -z "$VENV" ]; then
            env_error
        fi
        VENV="conda"
        shift
        ;;
    -d | --docker)
        if [ ! -z "$VENV" ]; then
            env_error
        fi
        VENV="docker"
        shift
        ;;
    -n | --no-download)
        DO_DOWNLOAD=0
        shift
        ;;
    --) shift;
        break
        ;;
  esac
done

if [ -z "$VENV" ]; then
    env_error
fi

if [ "$VENV" == "venv" ]; then
    virtualenv -p python3 .
    source ./bin/activate
elif [ "$VENV" == "conda" ]; then
    conda create -n ag python=3.10.12
    source ~/miniconda3/bin/activate ag
elif [ "$VENV" == "docker" ]; then
    docker build --no-cache . -t alphageometry_pt:latest
    docker run --name alphageo --gpus="all" -ti --rm --mount type=bind,src=.,target=/ag/ --entrypoint python alphageometry_pt:latest \
        common_folder_downloader.py --region cn-southwest-2 --app_token 82aaeb97-6bbb-4a9a-a164-07268a0a6d0b --bucket_name bucket-pangu-green-guiyang --path philipjg/pt_ckpt/
    mkdir results
    exit
fi

pip install -e .[download,torch]

if [ $DO_DOWNLOAD -eq 1 ]; then
    python common_folder_downloader.py --region cn-southwest-2 --app_token 82aaeb97-6bbb-4a9a-a164-07268a0a6d0b --bucket_name bucket-pangu-green-guiyang --path philipjg/pt_ckpt/
fi
