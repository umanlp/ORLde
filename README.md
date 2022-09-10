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
    - This folder contains the following files:
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
    - ORLde
        - This folder contains the data sets from the IGGSA-STEPS 2016 shared task, converted to json format:
        - shata14.train.json
        - shata14.val.json
        - steps16.test.json 
        



### Running the model:

Get predictions for the test data, using the trained model from:

#### run 1
```typescript
python bert_orl.py orl-1-predict-bert.conf
```

#### run 2
```typescript
python bert_orl.py orl-2-predict-bert.conf
```

#### run 3
```typescript
python bert_orl.py orl-3-predict-bert.conf
```


You can evaluate the predictions by running:

```typescript
python eval_predictions.py logfile_ORL_BERT_run_1.log 

python eval_predictions.py logfile_ORL_BERT_run_2.log 

python eval_predictions.py logfile_ORL_BERT_run_3.log 
```

### Training a new model:

You can train a new model on the training data and evaluate on the test set, using this script:

```typescript
python bert_orl.py 
```

