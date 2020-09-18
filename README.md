DAAT-CWS

Coupling Distant Annotation and Adversarial Training for Cross-Domain Chinese Word Segmentation

Paper accepted by ACL 2020

### Prerequisites
- python == 2.7
- tensorflow == 1.8.0

### Dataset

source domain dataset PKU and five distantly-annotated target datasets are put in  `data/datasets` directory

### Usage

Run `python train.py --tgt_train_path <tgt_train_path> --tgt_test_path <tgt_test_path>`


Note:

This code is based on the previous work by [chqiwang](https://github.com/chqiwang/convseg). Many thanks to [chqiwang](https://github.com/chqiwang/convseg).
The raw text of dataset used in our paper can be found at [CWS-NAACL2019](https://github.com/vatile/CWS-NAACL2019)
