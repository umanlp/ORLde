# ORLde
BERT model for German Opinion Role Labelling in Parliamentary Debates



Get predictions for the test data, using the trained model from:

# run 1
python bert_orl.py orl-1-predict-bert.conf

# run 2
python bert_orl.py orl-2-predict-bert.conf

# run 3
python bert_orl.py orl-3-predict-bert.conf


Evaluate the predictions:

for model 1
python eval_predictions.py logfile_ORL_BERT_run_1.log 

for model 2
python eval_predictions.py logfile_ORL_BERT_run_2.log 

for model 3
python eval_predictions.py logfile_ORL_BERT_run_3.log 



Train a new model on the training data and evaluate on test, using this model:

python bert_orl.py 
