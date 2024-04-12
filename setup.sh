virtualenv -p python3 .
source ./bin/activate

pip install -e .[download]

python common_folder_downloader.py --region cn-southwest-2 --app_token 82aaeb97-6bbb-4a9a-a164-07268a0a6d0b --bucket_name bucket-pangu-green-guiyang --path philipjg/pt_ckpt/
DATA=pt_ckpt
if [ ! -d "$DATA" ]; then
  cp -r /nfs/ainlp/math_agent/pt_ckpt .
fi
