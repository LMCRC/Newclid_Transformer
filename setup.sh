virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt
pip install -r requirements_internal.txt

DATA=pt_ckpt
if [ ! -d "$DATA" ]; then
  cp -r /nfs/ainlp/math_agent/pt_ckpt .
fi
