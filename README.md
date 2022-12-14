<h1 align="center">
<span>German Opinion Role Labelling (ORLde)</span>
</h1>

------------------------
## Repository Description

This repository contains the code for our 
BERT model for German Opinion Role Labelling in Parliamentary Debates.

The code can be used to replicate our results from the paper:

<a href="https://aclanthology.org/2022.konvens-1.13.pdf">Improved Opinion Role Labelling in Parliamentary Debates </a>


```
@inproceedings{bamberg-etal-2022-improved,
    title = "Improved Opinion Role Labelling in Parliamentary Debates",
    author = "Bamberg, Laura  and
    Rehbein, Ines  and
    Ponzetto, Simone",
    booktitle = "Proceedings of the 18th Conference on Natural Language Processing (KONVENS 2022)",
    month = "12--15 " # sep,
    year = "2022",
    address = "Potsdam, Germany",
    publisher = "KONVENS 2022 Organizers",
    url = "https://aclanthology.org/2022.konvens-1.13",
    pages = "110--120",
}
```

### Content:

- **code** 
      - models/ORL-1 (the 3 ORL single-task models)
      - bert_orl.py
      - eval_predictions.py
      - evaluation.py
      - helpers.py
      - orl-1-predict-bert.conf (config files for role prediction)
      - orl-2-predict-bert.conf
      - orl-3-predict-bert.conf
      - orl-train-bert.conf     (config file for training)

- **data**
    - ORLde (data sets from the IGGSA-STEPS 2016 shared task, converted to json format)
        - shata14.train.json
        - shata14.val.json
        - steps16.test.json 
        

The data has been kindly provided by the organisers of the IGGSA-STEPS 2016 Shared Task on Source and Target Extraction from Political Speeches
<a href="https://ids-pub.bsz-bw.de/files/5508/Ruppenhofer_Struss_Wiegand_Overview_of_the_IGGSA_2016.pdf">(pdf)</a>

```
@incollection{RuppenhoferStrussWiegand2016,
  author    = {Josef Ruppenhofer and Julia Maria Stru{\"s} and Michael Wiegand},
  title     = {Overview of the IGGSA 2016 Shared Task on Source and Target Extraction from Political Speeches},
  series    = {IGGSA Shared Task on Source and Target Extraction from Political Speeches},
  editor    = {Josef Ruppenhofer and Julia Maria Stru{\"s} and Michael Wiegand},
  publisher = {Ruhr-Universit{\"a}t Bochum},
  address   = {Bochum},
  issn      = {2190-0949},
  url       = {https://nbn-resolving.org/urn:nbn:de:bsz:mh39-55086},
  pages     = {1 -- 9},
  year      = {2016}, 
}
```
<a href="https://iggsasharedtask2016.github.io/welcome.html">Shared Task website</a>

------------------------

### Running the model

Get predictions for the test data, using the trained model from:

##### run 1:
```typescript
python bert_orl.py orl-1-predict-bert.conf
```

##### run 2:
```typescript
python bert_orl.py orl-2-predict-bert.conf
```

##### run 3:
```typescript
python bert_orl.py orl-3-predict-bert.conf
```


You can evaluate the predictions by running:

```typescript
python eval_predictions.py logfile_ORL_BERT_run_1.log 

python eval_predictions.py logfile_ORL_BERT_run_2.log 

python eval_predictions.py logfile_ORL_BERT_run_3.log 
```

### Training a new model

You can train a new model on the training data and evaluate on the test set, using this script:

```typescript
python bert_orl.py 
```
If you want to change the model parameters or input/output path, you need to change the config file (
orl-train-bert.conf)

