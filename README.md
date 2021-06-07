# Discourse Probing

## About

In this paper, we introduce document-level discourse probing to evaluate the ability of pretrained LMs to capture document-level relations. We experiment with 7 pretrained LMs,
4 languages, and 7 discourse probing tasks, and find BART to be overall the best model at capturing discourse — but only in its encoder.

## Paper
Fajri Koto, Jey Han Lau, and Timothy Baldwin. [_Discourse Probing of Pretrained Language Models_](https://www.aclweb.org/anthology/2021.naacl-main.301/). 
In Proceedings of the 20th Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2021), Mexico (virtual). 

## Discourse Probing Tasks
* Our code is based on Python3 and huggingface Pytorch framework. All libs is listed in `requirements.txt`. 
* Please run `pip install -r requirements.txt`

### 1. Next sentence prediction
* Similar to the next sentence prediction (NSP) objective in BERT pretraining, but here we frame it as a 4-way classification task, with one positive and 3
negative candidates for the next sentence
* Folder: `nsp_choice`. To run the code you can see the examples at `run.sh`.
* Data for EN, ES, DE, and ZH, are provided at `nsp_choice/data/`.

### 2. Sentence ordering
* We shuffle 3–7 sentences and attempt to reproduce the original order.
* Folder: `ordering`. To run the code you can see the examples at `run.sh`.
* Data for EN, ES, DE, and ZH, are provided at `ordering/data/`.

### 3. Discourse connective prediction
* Given two sentences/clauses, the task is to identify an appropriate discourse marker, such as "_while", "and_".
* Folder: `dissent`. To run the code you can see the examples at `run.sh`.
* Data for EN and DE are provided at `dissent/data/`. There is no data for ES.
* Some samples of ZH data are provided. Due to the re-distribution policy, you first need to request the [CDTB](https://www.aclweb.org/anthology/D14-1224/) to the related author.
You can extract the data by running `rst/prepare_data/extract_chinese_dtb.ipynb`

### 4-5. RST nuclearity and relation prediction.
* Nuclearity prediction: For a given ordered pairing of (potentially complex) EDUs which are connected by an unspecified relation, predict the nucleus/satellite status of each.
* Relation prediction: For a given ordered pairing of (potentially complex) EDUs which are connected by an unspecified relation, predict the relation that holds between them.
* Folder: `rst`. To run the code you can see the examples at `run_nuc.sh` and `run_rel.sh`.
* Some samples of EN, ES, DE, and ZH are provided. For ES, DE, and ZH, you can use: `rst/prepare_data/extract_chinese_dtb.ipynb`, `rst/prepare_data/extract_german_dtb.ipynb`, `rst/prepare_data/extract_spanish_dtb.ipynb` to extract the data (after downloading the related Discourse Tree Bank).
There is no code provided for extracting EN data.

### 6. RST elementary discourse unit (EDU) segmentation. 
* Chunk a concatenated sequence of EDUs into its component EDUs.
* Folder: `segment`. To run the code you can see the examples at `run.sh`.
* Some samples of EN, ES, DE, and ZH are provided. For ES, DE, and ZH, you can use: `extract_chinese_dtb.ipynb`, `extract_german_dtb.ipynb`, `extract_spanish_dtb.ipynb` to extract the data (after downloading the related Discourse Tree Bank).
There is no code provided for extracting EN data.

### 7. Cloze story test. 
* Given a 4-sentence story context, pick the best ending from two possible options ([Mostafazadeh et al., 2016](https://www.aclweb.org/anthology/N16-1098/)).
* Folder: `cloze`. To run the code you can see the examples at `run.sh`.
* This probing task is only for EN. First, please request the data to Mostafazadeh et al., 2016, and use `cloze/prepare_data.ipynb` to prepare the data. Some samples are provided at folder `cloze/data`

## Post Experiments

After running all the experiments, we provide some post-processing codes:
* `post_process.ipynb`: to extract mean and standard deviation of all experiments from 3 different runs.
* `plot_across_model.ipynb`: to create Figure 2 in the paper.
* `plot_across_langauges.ipynb`: to create Figure 3 in the paper.
