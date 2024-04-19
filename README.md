
# Solving Olympiad Geometry without Human Demonstrations


This repository contains the code necessary to install and run
DDAR and a version of AlphaGeometry using pyTorch to implement the LM

It should be able to reproduce results of DDAR and AlphaGeometry,
the two geometry theorem provers
introduced in the [Nature 2024](https://www.nature.com/articles/s41586-023-06747-5) paper:

*<center>"Solving Olympiad Geometry without Human Demonstrations".</center>*


</br>


<center>
<img alt="fig1" width="800px" src="fig1.svg">
</center>


## Dependencies

For the instructions presented below, we use Python 3.10.12.

It likely will work with other Python versions (at least >= 3.10), but is untested.

Required pip package version numbers are

```
sentencepiece==0.1.99
torch==2.2.2
boto3==1.33.13

```
Later versions may work, but are untested.

Our code depends on `geosolver v1.1.0`, which is another internal package
and not registered with `pip`/pypi.

Geosolver v1.1.0 is available at `https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/geosolver/-/tree/v1.1.0?ref_type=tags` .

This will automatically be installed with the `setup.sh` installation script, or by using
the `[geosolver]` option when installing manually.

Note that you will require login details or a valid token to clone the geosolver repository.

Note that one can still run the DDAR solver
without the `torch` and `sentencepiece` dependencies.

## Install dependencies, download weights and vocabulary.

### Fully automated

The easiest way to install AlphaGeometry is through the provided `setup.sh` script.

It allows to install either via `virtualenv`, `conda` or `docker`:
```shell
INSTALL WITH VIRTUALENV:
bash ./setup.sh --venv

INSTALL WITH CONDA
bash ./setup.sh --conda

INSTALL WITH DOCKER
bash ./setup.sh --docker
```

### Manually -- virtualenv/conda

Installation is done in a virtual environment.

For a python venv environment, first run

```shell
virtualenv -p python3 .
source ./bin/activate
```

For a conda environment, run
```shell
conda create -n ag python=3.10.12
conda activate ag
```

Then, install alphageo and its dependencies:
```shell
pip install -e .[download,torch,geosolver]
```

`download` installs boto3, which is needed to download model weights and vocabulary files.\
`torch` installs pyTorch 2.2.2 .\
`geosolver` installs the internal geosolver v1.1.0 from `https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/geosolver/-/tree/v1.1.0?ref_type=tags`.\

If you are participating in active development, you can install recommended dev features with the `[dev]` tag.

If any of these requirements are already installed on the host system (in the virtual environment), they can be omitted.

To get model weights and tokenizer vocab, and save in pt_ckpt

```shell
python common_folder_downloader.py --region cn-southwest-2 --app_token 82aaeb97-6bbb-4a9a-a164-07268a0a6d0b --bucket_name bucket-pangu-green-guiyang --path philipjg/pt_ckpt/
```

### Manually -- docker

To install alphageo as a docker image, first manually clone the `geosolver v1.1.0` dependency:
```shell
git clone --single-branch --branch v1.1.0 https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/geosolver.git
```

Then clean-build the docker:
```shell
docker build --no-cache . -t alphageometry_pt:latest
```

Get model weights and tokenizer vocab:
```shell
docker run --name alphageo --gpus="all" -ti --rm --mount type=bind,src=.,target=/ag/ --entrypoint python alphageometry_pt:latest \
common_folder_downloader.py --region cn-southwest-2 --app_token 82aaeb97-6bbb-4a9a-a164-07268a0a6d0b --bucket_name bucket-pangu-green-guiyang --path philipjg/pt_ckpt/
```

Optionally, remove the geosolver folder (it has been installed as a dependency inside the docker image, and is not required any longer)
```shell
rm -rf ./geosolver/
```

## Running alphageo

To run alphageo in either virtualenv or conda (whichever was selected, see above), first activate the respective environment:
```bash
source ./bin/activate
```
or
```bash
conda activate ag
```

Then simply run `alphageo` with parameters for your run
```
alphageo --problems_file [...]
```

To run alphageo with docker (if build above), make sure you bind the local `problems_datasets`, `pt_ckpt` and `results` folders.\
`alphageo` is already set as the docker's entrypoint, so just pass your parameters to the run command:
```bash
docker run --name alphageo --gpus="all" -ti --rm --mount type=bind,src="./pt_ckpt/",target=/ag/pt_ckpt/ --mount type=bind,src="./problems_datasets/",target=/ag/problems_datasets/ --mount type=bind,src="./results/",target=/ag/results/ --problems_file [...]
```

In the examples below, we assume `alphageo` is run from a virutalenv or conda environment.

### Running DDAR

The script loads a problem by reading a list of problems
from a text file and solves the specific problem in the list according
to its name. We pass these two pieces of information through the flags
`--problems_file` and `--problem`.
We use `--solver-only` to indicate that we want to use the DDAR solver alone, without the LM.

Below we show this solver solving IMO 2000 P1:

```shell
alphageo \
  --problems_file problems_datasets/imo_ag_30.txt \
  --problem translated_imo_2000_p1 \
  --solver-only \
  --logging
```

Expect the following output

```shell
INFO:root:translated_imo_2000_p1
INFO:root:a b = segment a b; g1 = on_tline g1 a a b; g2 = on_tline g2 b b a; m = on_circle m g1 a, on_circle m g2 b; n = on_circle n g1 a, on_circle n g2 b; c = on_pline c m a b, on_circle c g1 a; d = on_pline d m a b, on_circle d g2 b; e = on_line e a c, on_line e b d; p = on_line p a n, on_line p c d; q = on_line q b n, on_line q c d ? cong e p e q
INFO:root:Depth 1/1000 time = 0.6569983959197998
INFO:root:Depth 2/1000 time = 2.1829771995544434
INFO:root:Depth 3/1000 time = 2.920427083969116
INFO:root:Depth 4/1000 time = 3.995807409286499
INFO:root:Depth 5/1000 time = 5.559977293014526
INFO:root:Solved.
INFO:root:
==========================
 * From theorem premises:
A B G1 G2 M N C D E P Q : Points
AG_1 ⟂ AB [00]
BA ⟂ G_2B [01]
G_1M = G_1A [02]
G_2M = G_2B [03]
G_2N = G_2B [04]
G_1N = G_1A [05]
CM ∥ AB [06]
G_1C = G_1A [07]
∠NAC = ∠NAC [08]
G_2D = G_2B [09]
DM ∥ AB [10]
∠DBN = ∠DBN [11]
E,C,A are collinear [12]
E,D,B are collinear [13]
P,C,D are collinear [14]
P,A,N are collinear [15]
C,D,Q are collinear [16]
N,B,Q are collinear [17]
DQ:QM = DQ:QM [18]

 * Auxiliary Constructions:
: Points


 * Proof steps:
001. G_2N = G_2B [04] & G_2D = G_2B [09] ⇒  G_2 is the circumcenter of \Delta BND [19]
002. G_2 is the circumcenter of \Delta BND [19] & G_2B ⟂ BA [01] ⇒  ∠DBA = ∠DNB [20]
003. DM ∥ AB [10] & CM ∥ AB [06] ⇒  MC ∥ MD [21]

[...snip...]

047. E,A,C are collinear [12] & C,D,Q are collinear [16] & E,D,B are collinear [13] & ∠ENA = ∠EBA [49] & ∠ANE = ∠PEA [64] & DM ∥ AB [10] & C,D,M are collinear [22] ⇒  ∠PEC = ∠QDE [65]
048. ∠PCE = ∠QED [53] & ∠PEC = ∠QDE [65] (Similar Triangles)⇒  PE:DQ = EC:ED [66]
049. EC:ED = EQ:DQ [52] & PE:DQ = EC:ED [66] ⇒  PE:DQ = EQ:DQ [67]
050. PE:DQ = EQ:DQ [67] & DQ:QM = DQ:QM [18] ⇒  EQ = PE
==========================

INFO:root:Solution written to results/translated_imo_2000_p1/proof_steps.txt.
```

TIP: `results/translated_imo_2000_p1/` will also contain a `proof_figure.png`, visualising the found proof.

The output first includes a list of relevant premises that it uses,
and then proof steps that gradually build up the proof.
All predicates are numbered to track how they are derived
from the premises, and to show that the proof is fully justified.

Running on all problems in `imo_ag_30.txt` will yield solutions to
14 of them, as reported in Table 1 in the original AlphaGeometry paper.

## Run AlphaGeometry:

As a simple example, we load `--problem=orthocenter`
from `--problems_file=problems_datasets/examples.txt`.
This time, we don't pass the `--solver-only` argument, meaning we will make use of the LM
to generate auxiliary constructions when DDAR fails.

TIP: There are a number of optional arguments we can provide to `alphageo` to guide
LM generation and solution exploration. Use `alphageo --help` to see all available
parameters and their default values.

```shell
alphageo \
--problems_file problems_datasets/examples.txt \
--problem orthocenter \
--logging
```

Expect the following output:

```shell
INFO:root:orthocenter
INFO:root:a b c = triangle; d = on_tline b a c, on_tline c a b ? perp a d b c
INFO:root:Depth 1/1000 time = 0.007330179214477539
INFO:root:Depth 2/1000 time = 0.0073146820068359375
INFO:root:Solver failed to solve the problem.
INFO:root:Depth 0. There are 1 nodes to expand:
INFO:root:{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c {F1} x00
INFO:root:Decoding from {S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c {F1} x00
INFO:root:LM output 1: e : C a c e 02 C b d e 03 ; (score: -1.6525408374450157)
INFO:root:LM output 2: e : D a b c e 02 D a c b e 03 ; (score: -1.8277973403035555)
INFO:root:Trying LM output (score=-1.6525408374450157): "e : C a c e 02 C b d e 03 ;"
INFO:root:Translation: "e = on_line e a c, on_line e b d"

INFO:root:Solving: "a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c"
INFO:root:orthocenter
INFO:root:a b c = triangle a b c; d = on_tline d b a c, on_tline d c a b; e = on_line e a c, on_line e b d ? perp a d b c
INFO:root:Depth 1/1000 time = 0.011323690414428711
INFO:root:Depth 2/1000 time = 0.013859272003173828
INFO:root:Depth 3/1000 time = 0.016915082931518555
INFO:root:Solved.
INFO:root:
==========================
 * From theorem premises:
A B C D : Points
BD ⟂ AC [00]
CD ⟂ AB [01]

 * Auxiliary Constructions:
E : Points
E,B,D are collinear [02]
C,A,E are collinear [03]

 * Proof steps:
001. B,E,D are collinear [02] & C,A,E are collinear [03] & BD ⟂ AC [00] ⇒  ∠BEA = ∠CED [04]
002. B,E,D are collinear [02] & C,A,E are collinear [03] & BD ⟂ AC [00] ⇒  ∠BEC = ∠AED [05]
003. B,E,D are collinear [02] & BD ⟂ AC [00] ⇒  EB ⟂ CA [06]
004. CD ⟂ AB [01] & EB ⟂ CA [06] ⇒  ∠(CD-EB) = ∠BAC [07]
005. C,A,E are collinear [03] & E,B,D are collinear [02] & ∠(CD-EB) = ∠BAC [07] ⇒  ∠BAE = ∠CDE [08]
006. ∠BEA = ∠CED [04] & ∠BAE = ∠CDE [08] (Similar Triangles)⇒  BE:CE = AE:ED [09]
007. BE:CE = AE:ED [09] & ∠BEC = ∠AED [05] (Similar Triangles)⇒  ∠BCE = ∠ADE [10]
008. BE:CE = AE:ED [09] & ∠BEC = ∠AED [05] (Similar Triangles)⇒  ∠EBC = ∠EAD [11]
009. ∠BCE = ∠ADE [10] & C,A,E are collinear [03] & E,B,D are collinear [02] & ∠EBC = ∠EAD [11] ⇒  AD ⟂ BC
==========================

INFO:root:Solution written to results/orthocenter/proof_steps.txt.
```

NOTE: Point `H` is automatically renamed to `D`,
as the LM is trained on synthetic problems
where the points are named alphabetically, and so it expects
the same during test time.

NOTE: This implementation of AlphaGeometry, is missing all all optimizations that are dependent on
internal infrastructure of the original authors (DeepMind), e.g., parallelized model inference on
multi GPUs, parallelized DDAR on multiple CPUs, parallel execution of LM and DDAR, shared pool of
CPU workers across different problems, etc. They also removed some memory/speed optimizations and code
abstractions in favor of "code clarity".

As can be seen in the output, initially DDAR failed to solve the problem.
The LM proposes two auxiliary constructions (because of default parameter `--batch_size=2`):\
`e : C a c e 02 C b d e 03 ; (score: -1.6525408374450157)`
`e : D a b c e 02 D a c b e 03 ; (score: -1.8277973403035555)`

* `e = on_line e a c, on_line e b d`, i.e.,
`E` is the intersection of `AC` and `BD`.
This construction has the highest score (`-1.6525...`) of the generated solution, and is explored first..

DDAR re-runs with the proposed auxiliary construction, and finds the solution right away.
The proof search therefore terminates and there is no second iteration.

## Results

Before attempting to reproduce the AlphaGeometry numbers in our paper or pushing to the reposity,
please make sure to pass all tests in the prepared test suite:

```shell
pytest

=================== test session starts ====================
platform linux -- Python 3.10.14, pytest-8.1.1, pluggy-1.4.0
rootdir: /nfs/ainlp/philipjg/math_agent/alphageometry
configfile: pyproject.toml
plugins: mock-3.14.0, check-2.3.1, cov-5.0.0
collected 13 items

tests/alphageometry_test.py ...                       [ 23%]
tests/beam_queue_test.py .                            [ 30%]
tests/translate_test.py ........                      [100%]

==================== 13 passed in 5.48s ====================
```


Then, pass the corresponding values for `--problem_file` (column)
and `--mode` (row), and
iterate on all problems to obtain the following results:

<center>

<b>Number of solved problems:</b>

|          | `imo_ag_30.txt`  | `jgex_ag_231.txt` |
|----------|------------------|-------------------|
| `ddar`   | 14               | 198               |
| `alphageometry`     | 25               | 228               |

</center>

## Source code description

Files in this repository include python modules/scripts to run the solvers and
resource files necessary for the script to execute. We listed below
each of them and their description.

| File name              | Description                                                                                      |
|------------------------|--------------------------------------------------------------------------------------------------|
| `common_folder_downloader.py`          | Used to download model weights and vocabulary from S3.                           |
| `convert_ag_to_pt.py`  | Used to convert original AG (Meliad/Flax) model parameters to pyTorch. NOTE: Requires original DeepMind AG installation.              |
| `./src/alphageo/`        | |
| `alphageometry.py`     | Implements actual runs of DDAR/AG             .                                                  |
| `cli.py `              | Provides command line arguments and defaults.                                                    |
| `inference.py`         | Implements LM inference methods, such as beam search.                                            |
| `__main__.py`          | Provides the main entrypoint for alphageo. Performs required setups and calls `alphageometry.py` |
| `model.py`             | Implements pyTorch version of AG's LM.                                                           |
| `optional_imports.py`  | Used to handle optional modules.                                                                 |
| `translate.py`         | Translates from LM outputs to DDAR-compatible strings.                                           |
| `./test/*_test.py`     | Test suite files for regression testing of AG components.                                        |


Resource files:

| Resource file name     | Description                                           |
|------------------------|-------------------------------------------------------|
| `./pt_ckpt/` | |
| `cfg.sav` | Pretrained LM configuration file.                                  |
| `params.sav` | Pretrained LM parameters.                                       |
| `vocab.model` | Pretrained sentencepiece tokenizer model.                      |
| `vocab.vocab` | Sentencepiece vocabulary.                                      |
| `./problems_datasets/` | |
| `examples.txt`            | Geometric problem dataset with some test problems. |
| `imo_ag_30.txt`        | Problems in IMO-AG-30.                                |
| `jgex_ag_231.txt`      | Problems in JGEX-AG-231.                              |



## Citing this work

```bibtex
@Article{AlphaGeometryTrinh2024,
  author  = {Trinh, Trieu and Wu, Yuhuai and Le, Quoc and He, He and Luong, Thang},
  journal = {Nature},
  title   = {Solving Olympiad Geometry without Human Demonstrations},
  year    = {2024},
  doi     = {10.1038/s41586-023-06747-5}
}
```

## Acknowledgements

This research is a collaboration between the Google Brain team
(now Google Deepmind) and
the Computer Science Department of New York University.
We thank Rif A. Saurous, Denny Zhou, Christian Szegedy, Delesley Hutchins,
Thomas Kipf, Hieu Pham, Petar Veličković, Debidatta Dwibedi,
Kyunghyun Cho, Lerrel Pinto, Alfredo Canziani,
Thomas Wies, He He’s research group,
Evan Chen (the USA’s IMO team coach),
Mirek Olsak, Patrik Bak,
and all three Nature's referees for their help and support.

The code of AlphaGeometry communicates with and/or references the following
separate libraries and packages:

*   [Abseil](https://github.com/abseil/abseil-py)
*   [JAX](https://github.com/google/jax/)
*   [matplotlib](https://matplotlib.org/)
*   [NumPy](https://numpy.org)
*   [SciPy](https://scipy.org)
*   [TensorFlow](https://github.com/tensorflow/tensorflow)
*   [Meliad](https://github.com/google-research/meliad)
*   [Flax](https://github.com/google/flax)
*   [Gin](https://github.com/google/gin-config)
*   [T5](https://github.com/google-research/text-to-text-transfer-transformer)
*   [SentencePiece](https://github.com/google/sentencepiece)



We thank all their contributors and maintainers!


## Disclaimer

This is not an officially supported Google product.

This research code is provided "as-is" to the broader research community.
Google does not promise to maintain or otherwise support this code in any way.

## Code License

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

## Model Parameters License

The AlphaGeometry checkpoints and vocabulary are made available
under the terms of the Creative Commons Attribution 4.0
International (CC BY 4.0) license.
You can find details at:
https://creativecommons.org/licenses/by/4.0/legalcode

