## A Dataset for Hyper-Relational Extraction and a Cube-Filling Approach

[//]: # ([![PWC]&#40;https://img.shields.io/badge/PapersWithCode-Benchmark-%232cafb1&#41;]&#40;https://paperswithcode.com/paper/relationprompt-leveraging-prompts-to-generate&#41;)

[//]: # ([![Colab]&#40;https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00&#41;]&#40;https://colab.research.google.com/drive/18lrKD30kxEUolQ61o5nzUJM0rvWgpbFK?usp=sharing&#41;)

[//]: # ([![Jupyter]&#40;https://img.shields.io/badge/Jupyter-Notebook%20Demo-important&#41;]&#40;https://github.com/declare-lab/RelationPrompt/blob/main/demo.ipynb&#41;)

This repository implements our [EMNLP 2022 research paper](https://arxiv.org/abs/2211.10018).

HyperRED is a dataset for the new task of hyper-relational extraction, which extracts relation triplets together with
qualifier information such as time, quantity or location.
For example, the relation triplet (Leonard Parker, Educated At, Harvard University) can be factually enriched by
including the qualifier (End Time, 1967).
HyperRED contains 44k sentences with 62 relation types and 44 qualifier types.
Inspired by table-filling approaches for relation extraction, we propose CubeRE, a cube-filling model which explicitly
considers the interaction between relation triplets and qualifiers.

[//]: # (![diagram]&#40;https://github.com/declare-lab/RelationPrompt/releases/download/v1.0.0/diagram.png&#41;)

### Setup

Install Python Environment

```
conda create -n cube python=3.7 -y
conda activate cube
pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

Download HyperRED Dataset (Available on [Huggingface Datasets](https://huggingface.co/datasets/declare-lab/HyperRED))

```
python data_process.py download_data data/hyperred/
python data_process.py process_many data/hyperred/ data/processed/
```

### Data Exploration

```
from data_process import Data

path = "data/hyperred/train.json"
data = Data.load(path)

for s in data.sents[:3]:
    print()
    print(s.tokens)
    for r in s.relations:
        print(r.head, r.label, r.tail)
        for q in r.qualifiers:
            print(q.label, q.span)
```

### Model Training

```
python training.py \
--save_dir ckpt/cube_prune_20_seed_0 \
--seed 0 \
--data_dir data/processed \
--prune_topk 20 \
--config_file config.yml
```

### Research Citation

If the code is useful for your research project, we appreciate if you cite the
following [paper](https://arxiv.org/abs/2211.10018):

```
@inproceedings{chia-etal-2022-hyperred,
    title = "A Dataset for Hyper-Relational Extraction and a Cube-Filling Approach",
    author = "Chia, Yew Ken and Bing, Lidong and Aljunied, Sharifah Mahani and Si, Luo and Poria, Soujanya",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    url = "https://arxiv.org/abs/2211.10018",
    doi = "https://arxiv.org/abs/2203.09101",
}
```
