virtualenv -p python3 .
source ./bin/activate

pip install --require-hashes -r requirements.txt
pip install -r requirements_internal.txt

DATA=ag_ckpt_vocab
if [ ! -d "$DATA" ]; then
  gdown --folder https://bit.ly/alphageometry
fi

MELIAD_PATH=meliad_lib/meliad
if [ ! -d "$MELIAD_PATH" ]; then
    mkdir -p $MELIAD_PATH
    git clone https://github.com/google-research/meliad $MELIAD_PATH
fi
