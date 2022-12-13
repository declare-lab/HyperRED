## A Dataset for Hyper-Relational Extraction and a Cube-Filling Approach

[![HD](https://img.shields.io/badge/HuggingFace-Datasets-blue)](https://huggingface.co/datasets/declare-lab/HyperRED)
[![PWC](https://img.shields.io/badge/PapersWithCode-Benchmark-%232cafb1)](https://paperswithcode.com/sota/hyper-relational-extraction-on-hyperred)
[![Colab](https://img.shields.io/badge/Colab-Code%20Demo-%23fe9f00)](https://colab.research.google.com/drive/1R3nDZ278vUlPrjfJPoTB7fFA1JFN8h5-?usp=sharing)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook%20Demo-important)](https://github.com/declare-lab/HyperRED/blob/master/demo.ipynb)

This repository implements our [EMNLP 2022 research paper](https://arxiv.org/abs/2211.10018).

![diagram](https://github.com/declare-lab/HyperRED/releases/download/v1.0.0/data.png)

HyperRED is a dataset for the new task of hyper-relational extraction, which extracts relation triplets together with
qualifier information such as time, quantity or location.
For example, the relation triplet (Leonard Parker, Educated At, Harvard University) can be factually enriched by
including the qualifier (End Time, 1967).
HyperRED contains 44k sentences with 62 relation types and 44 qualifier types.
Inspired by table-filling approaches for relation extraction, we propose CubeRE, a cube-filling model which explicitly
considers the interaction between relation triplets and qualifiers.

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

### Data Fields

- **tokens:** Sentence text tokens.
- **entities:** List of each entity span. The span indices correspond to each token in the space-separated text (
  inclusive-start and exclusive-end index)
- **relations:** List of each relationship label between the head and tail entity spans. Each relation contains a list
  of qualifiers where each qualifier has the value entity span and qualifier label.

### Data Example

An example instance of the dataset is shown below:

```
{              
  "tokens": ['Acadia', 'University', 'is', 'a', 'predominantly', 'undergraduate', 'university', 'located', 'in', 'Wolfville', ',', 'Nova', 'Scotia', ',', 'Canada', 'with', 'some', 'graduate', 'programs', 'at', 'the', 'master', "'", 's', 'level', 'and', 'one', 'at', 'the', 'doctoral', 'level', '.'],
  "entities": [
    {'span': (0, 2), 'label': 'Entity'},
    {'span': (9, 13), 'label': 'Entity'},
    {'span': (14, 15), 'label': 'Entity'},
  ],
  "relations": [
    {
      "head": [0, 2],
      "tail": [9, 13],
      "label": "headquarters location",
      "qualifiers": [
        {"span": [14, 15], "label": "country"}
      ]
    }
  ], 
}
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

### Model Prediction

You can download and extract the pre-trained
weights [here](https://github.com/declare-lab/HyperRED/releases/download/v1.0.0/cube_model.zip)

```
from prediction import run_predict

texts = [
    "Leonard Parker received his PhD from Harvard University in 1967 .",
    "Szewczyk played 37 times for Poland, scoring 3 goals .",
]
preds = run_predict(texts, path_checkpoint="cube_model")
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
}
```
