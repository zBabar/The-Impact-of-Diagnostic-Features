# The-Impact-of-Diagnostic-Features

We employed R2Gen code as base code for report generation task. However, we have prepared our own features extraction and preparation pipeline

Here are the links to the paper and code

```
@inproceedings{chen2020generating,
  title={Generating Radiology Reports via Memory-driven Transformer},
  author={Chen, Zhihong and Yang, Lifeng and Wang, Maosong and Shen, Zhihong and Yu, Kai and He, Qiyuan},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={8111--8123},
  year={2020}
}

```

Code:

https://github.com/cuhksz-nlp/R2Gen

### Data:

In order to download data please refer the the data folder and follow the instructions

### Feature Extraction

Primarily, we extract features either from CheXNet or finetuning the models based on the CheXBert labels and then extracting the features. Please refer to feature_extraction folder

### Feature Appending

After extracting both the visual and diganostic freatures, we append the diagnostic features with visual featues. Please refer to the prep_data folder.
