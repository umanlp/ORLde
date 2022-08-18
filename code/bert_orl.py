import torch
import numpy as np
import logger
import logging
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split, SequentialSampler
from transformers import Trainer, AdamW, get_linear_schedule_with_warmup
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import get_dataset_config_names
from seqeval.metrics import f1_score
from collections import defaultdict
from datasets import DatasetDict, load_dataset
import torch.optim as optim
import pandas as pd
import helpers, evaluation
import datetime
import json
import random
import ast
import os, sys
import configparser


config_parser = configparser.ConfigParser()
config_parser.read(sys.argv[1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainfile = config_parser['BASE']['filepath_orl_train']
devfile   = config_parser['BASE']['filepath_orl_dev']
testfile  = config_parser['BASE']['filepath_orl_test']

data = load_dataset('json', data_files={'train': trainfile, 'validation': devfile, 'test': testfile })

task = config_parser['BASE']['task']
model_name = config_parser['BASE']['bert_model']
model_abbr = config_parser['BASE']['model_abbr']
model_name_str = f"{model_name}-finetuned-{task}"
model_save_path = 'models/' + model_abbr + "-" + task + "/"
train_data_type = config_parser['TRAIN']['data_type']
test_data_type = config_parser['TEST']['data_type']

labels = ["[PAD]", "[UNK]", "O", "B-Source", "I-Source", "B-Target", "I-Target", "X", "Speaker"]


label2index, index2label = {}, {}
for i, item in enumerate(labels):
    label2index[item] = i
    index2label[i] = item


model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2index)).to("cuda")

bert_tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_and_align_labels(examples):
    tokenized_inputs = bert_tokenizer(examples["words"],
                                      truncation=True,
                                      padding='max_length',
                                      max_length=120,
                                      is_split_into_words=True)
    labels = []; predicates = []

    for idx, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []; pred_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
                pred_ids.append(0)
            else:
                label_ids.append(label[word_idx])
                if examples["token_type_ids"][idx][word_idx] == 1:
                    pred_ids.append(1)
                else:
                    pred_ids.append(0)
            previous_word_idx = word_idx
        labels.append(label_ids)
        predicates.append(pred_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["token_type_ids"] = predicates
    return tokenized_inputs



def encode_dataset(corpus):
    return corpus.map(tokenize_and_align_labels, batched=True, remove_columns=['words', 'tags'])



def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2label[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2label[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list


data_encoded = encode_dataset(data)  


#############
### Load data
tr_input_ids, tr_attention_masks, tr_label_ids, tr_token_type_ids, tr_seq_lengths = helpers.load_input(data_encoded["train"])
train_dataset = TensorDataset(tr_input_ids, tr_attention_masks, tr_label_ids, tr_seq_lengths, tr_token_type_ids)
dev_input_ids, dev_attention_masks, dev_label_ids, dev_token_type_ids, dev_seq_lengths = helpers.load_input(data_encoded["validation"])
val_dataset   = TensorDataset(dev_input_ids, dev_attention_masks, dev_label_ids, dev_seq_lengths, dev_token_type_ids)
te_input_ids, te_attention_masks, te_label_ids, te_token_type_ids, te_seq_lengths = helpers.load_input(data_encoded["test"])
test_dataset  = TensorDataset(te_input_ids, te_attention_masks, te_label_ids, te_seq_lengths, te_token_type_ids)


###############################
### Parameter settings:
### learning rate and optimizer
EPOCHS = int(config_parser['PARAM']['epochs'])
BATCH_SIZE = int(config_parser['PARAM']['batch_size'])
LEARNING_RATE = float(config_parser['PARAM']['learning_rate'])
EPS = float(config_parser['PARAM']['eps'])
SEED = int(config_parser['PARAM']['seed']) # seeds used for fold 1:42, 2:18, 3:5
START_EPOCH = 0
GRADIENT_CLIP = float(config_parser['PARAM']['gradient_clip'])
PRINT_INFO_EVERY = int(config_parser['PARAM']['print_info_every'])
OPTIMIZER = config_parser['PARAM']['optimizer']
NUM_WARMUP_STEPS = float(config_parser['PARAM']['num_warmup_steps'])
WEIGHT_DECAY = float(config_parser['PARAM']['weight_decay'])
loss_values = []
RUN = config_parser['PARAM']['run']


logging_steps = len(data_encoded["train"]) // BATCH_SIZE
model_name_str = f"{model_name}-finetuned-{task}"
train_logfile  = "train_" + train_data_type + "_" + model_abbr + "_run_" + str(RUN) + ".log"
test_logfile   = "logfile_" + test_data_type + "_" + model_abbr + "_run_" + str(RUN) + ".log"
pred_file      = "predictions_" + test_data_type + "_" + model_abbr + "_run_" + str(RUN) + ".txt"

file_name = train_data_type + "-" + task + "_train_" + model_abbr + "_run_" + str(RUN) + ".txt"

total_steps = len(data_encoded["train"]) * EPOCHS


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, 
                weight_decay=WEIGHT_DECAY, eps=EPS)

scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS,
                num_training_steps=total_steps)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



######################
### create data loader
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
val_sampler = RandomSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)



#################
### Training loop

if config_parser['BASE'].getboolean('train') == True:

  logging.info("Start training.")
  dict_source = {}
  dict_target = {}
  dict_source["source_tp"] = 0
  dict_source["source_fp"] = 0
  dict_source["source_fn"] = 0
  dict_target["target_tp"] = 0
  dict_target["target_fp"] = 0
  dict_target["target_fn"] = 0
  logging.info("=> ORL dictionaries created.")


  trainlogfile = open(train_logfile, 'w')
  trainlogfile.write("--------------------------------------------------------------------------------------------\n")

  for epoch_i in range(START_EPOCH+1, EPOCHS+1):
    logging.info("--------------------------------------------------------------------------------------------")
    logging.info("Epoch {:} / {:}".format(epoch_i, EPOCHS))

    start_time = datetime.datetime.now()
    logging.info("Start Time: %s ", start_time.strftime("%X"))
    trainlogfile.write(str(start_time) + "\n")
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_length = len(batch[3]) #neu!
        b_predicates = batch[4].to(device)
        
        model.zero_grad()

        outputs = model(b_input_ids, token_type_ids=b_predicates, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        optimizer.step()

        scheduler.step()

        if step % PRINT_INFO_EVERY == 0 and step != 0:
          logging.info("%s %s %s", step, len(train_dataloader), loss.item())



    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)

    logging.info("")
    logging.info("Average training loss: {0:.4f}".format(avg_train_loss))
    logging.info("")
    time = datetime.datetime.now()
    logging.info("Duration: ", time-start_time, "(h:mm:ss.microseconds)")

    trainlogfile.write("Average Training Loss: \n")
    trainlogfile.write(str(round(avg_train_loss, 3)) + "\n")
    trainlogfile.write("Duration:\t" + str(time-start_time))
    trainlogfile.write(" h:mm:ss.microseconds\n")

    ########## Validation ##########
    model.eval()
    total_sents = 0

    # Evaluate data for one epoch
    for batch in val_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels, b_len, b_predicates = batch

      # Telling the model not to compute or store gradients, saving memory and speeding up validation
      with torch.no_grad():
        # Forward pass, calculate logit predictions.
        outputs = model(b_input_ids, token_type_ids=b_predicates, attention_mask=b_input_mask)

      logits = outputs[0]
      output_vals = torch.softmax(logits, dim=-1)

      # Move class_probabilities and labels to CPU
      class_probabilities = output_vals.detach().cpu().numpy()
      argmax_indices = np.argmax(class_probabilities, axis=-1)

      label_ids = b_labels.to('cpu').numpy()
      seq_lengths = b_len.to('cpu').numpy()

      for ix in range(len(label_ids)):
        total_sents = total_sents +1

        # Store predictions and true labels
        pred_labels = [index2label[argmax_indices[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]
        gold_labels = []
        for g in range(len(label_ids[ix])):
            if label_ids[ix][g] != -100: gold_labels.append(index2label[label_ids[ix][g]])

        if len(pred_labels) != len(gold_labels):
            logging.info("Predictions not as long as gold")

        source_tp, source_fp, source_fn, target_tp, target_fp, target_fn, dict_source, dict_target = evaluation.orl_evaluation(gold_labels, pred_labels, dict_source, dict_target)
  
    source_recall, source_precision, source_f1, target_recall, target_precision, target_f1 = evaluation.orl_micro_average(dict_source, dict_target)

    trainlogfile.write("\nSource: \n")
    trainlogfile.write("TP: ")
    trainlogfile.write("TP:\t" + str(dict_source["source_tp"]) + ", ")
    trainlogfile.write("FP:\t" + str(dict_source["source_fp"]) + ", ")
    trainlogfile.write("FN: ")
    trainlogfile.write(str(dict_source["source_fn"]) + "\n")
    trainlogfile.write("Recall: ")
    trainlogfile.write(str(round(source_recall, 3)) + ", ")
    trainlogfile.write("Precision: ")
    trainlogfile.write(str(round(source_precision, 3)) + ", ")
    trainlogfile.write("F1: ")
    trainlogfile.write(str(round(source_f1, 3)) + "\n")

    trainlogfile.write("Target:\n")
    trainlogfile.write("TP:\n" + str(dict_target["target_tp"]) + ", ")
    trainlogfile.write("FP:\n" + str(dict_target["target_fp"]) + ", ")
    trainlogfile.write("FN:\n" + str(dict_target["target_fn"]) + "\n")
    trainlogfile.write("Recall:\t" + str(round(target_recall, 3)) + ", ")
    trainlogfile.write("Precision:\t" + str(round(target_precision, 3)) + ", ")
    trainlogfile.write("F1:\t" + str(round(target_f1, 3)) + "\n")


    # reset dictionary values
    dict_source["source_tp"] = 0
    dict_source["source_fp"] = 0
    dict_source["source_fn"] = 0
    dict_target["target_tp"] = 0
    dict_target["target_fp"] = 0
    dict_target["target_fn"] = 0


    if epoch_i >= 3:

      new_folder = train_data_type + "_Run_" + str(RUN) + "_Epochs_" + str(epoch_i) + "/"
      model_dir = model_save_path + train_data_type + "_" + model_name_str + "_" + new_folder
      model_info = model_abbr + "_" + str(RUN) + "_" + str(epoch_i)

      logging.info("SAVE MODEL TO %s", model_dir)
      helpers.save_model(model_dir, model_info, model, bert_tokenizer)

  trainlogfile.close()



###################
### Test loop

if config_parser['BASE'].getboolean('test') == True:
  logging.info("Writing logging info to %s", test_logfile)
  logging.info("Writing predictions to %s", pred_file)
  outputfile = open(test_logfile, 'w')
  outputfile.write(file_name + "\n")
  predfile = open(pred_file, 'w')
  predfile.write("WORD\tGOLD\tPRED\n")

  model.eval()
  total_sents = 0

  for batch in test_dataloader:
    # Add batch to GPU
    # Unpack the inputs from our dataloader
    t_input_ids, t_input_masks, t_labels, t_lengths, t_token_type_ids = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(t_input_ids, token_type_ids=t_token_type_ids, attention_mask=t_input_masks)

    logits = outputs[0]
    class_probabilities = torch.softmax(logits, dim=-1)

    # Move class_probabilities and labels to CPU
    class_probabilities = class_probabilities.detach().cpu().numpy()
    argmax_indices = np.argmax(class_probabilities, axis=-1)

    label_ids = t_labels.to('cpu').numpy()
    token_ids = t_token_type_ids.to('cpu').numpy()
    seq_lengths = t_lengths.to('cpu').numpy()


    for ix in range(len(label_ids)):
        total_sents += 1

        # Store predictions and true labels
        pred_labels = [index2label[argmax_indices[ix][p]] for p in range(len(label_ids[ix])) if label_ids[ix][p] != -100]
        gold_labels, token_labels = [], []
        for g in range(len(label_ids[ix])):
            if label_ids[ix][g] != -100: 
                gold_labels.append(index2label[label_ids[ix][g]])
                token_labels.append(token_ids[ix][g])

        if len(pred_labels) != len(gold_labels):
            logging.info("Predictions not as long as gold: %s", total_sents)

        text = bert_tokenizer.convert_ids_to_tokens(t_input_ids[ix], skip_special_tokens=False)
        clean_text = []
        for i in range(1, len(text)):
            if label_ids[ix][i] == -100:
                clean_text[-1] += text[i].replace('##', '')
            else:
                clean_text.append(text[i])

        clean_text[-1] = "Dummy"
        if len(clean_text) != len(pred_labels) or len(clean_text) != len(gold_labels):
            logging.info("ERROR: %s %s %s", len(clean_text), len(gold_labels), len(pred_labels))
        outputfile.write("\n" + str(total_sents) + "\n" + str(clean_text) + "\n")
        outputfile.write("GOLD: " + str(gold_labels) + "\nPRED: " + str(pred_labels) + "\nINDEX: " + str(token_labels))

        # writing predictions to file, one word per line
        for i in range(len(clean_text)):
            if token_labels[i] == 1:
                predfile.write("SE__" + clean_text[i] + "\t" + gold_labels[i] + "\t" + pred_labels[i] + "\n")
            else:
                predfile.write(clean_text[i] + "\t" + gold_labels[i] + "\t" + pred_labels[i] + "\n")
        predfile.write("\n")

  outputfile.close()
  predfile.close()
