# ConPer

Code and datasets for our paper [Persona-Guided Planning for Persona-Aware Story Generation](https://arxiv.org/pdf/2204.10703.pdf)

## 1. Environment Setup

python 3.7

pip 20.3.3

Install dependencies
```
pip install -r requirements.txt
```
## 2.Run
### Preparation

#### Download datasets
The preprocessed datasets can be obtained from this [link](https://drive.google.com/drive/u/0/folders/1MrcOc04waE13U-PXx5nkQRQiaB7Si5ON). 

You need to put the preprocessed data in `data/`.

#### Download fine-tuned model

The fine-tuned model can be obatined from this [link](https://drive.google.com/drive/u/0/folders/13p0TZocDWLfnUQLcO_78Zo01q57-xjWE)

You need to put the fine-tuned checkpoint in `results/`.

### Train

To train a model, you can run the following command, where `0` denotes GPU_ID.

```
bash scripts/train.sh 0
```
### Generate

To generate stories, you can run the following command, where `0` denotes GPU_ID.

```
bash scripts/generate.sh 0

main arguments:

--ckpt_path: path of the fine-tuned checkpoint
--output_path: path of the generation result
```




