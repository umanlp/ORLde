import torch
import os



def compute_metrics(eval_pred):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}


def load_input(data):
    if torch.cuda.is_available():    
        LongTensor = torch.cuda.LongTensor 
    else:
        LongTensor = torch.LongTensor
        
    seq_lengths = [len(i) for i in data['input_ids']]    
    input_ids = LongTensor(data['input_ids'])
    attention_masks = LongTensor(data['attention_mask'])
    label_ids = LongTensor(data['labels'])
    seq_lengths = LongTensor(seq_lengths)
    token_type_ids = LongTensor(data['token_type_ids'])
    
    return input_ids, attention_masks, label_ids, token_type_ids, seq_lengths




def save_model(output_dir, model_info, model, tokenizer):
    # Create output directory if needed
    out_path = output_dir + '/' + model_info
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + '/' + model_info):
        os.makedirs(output_dir + '/' + model_info)

    print("Saving model to %s" % out_path)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)



def save_adapter(output_dir, model_info, model, task_name, tokenizer):
    # Create output directory if needed
    out_path = output_dir + '/' + model_info
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + '/' + model_info):
        os.makedirs(output_dir + '/' + model_info)

    print("Saving adapter to %s" % out_path)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_adapter(out_path, task_name)
    tokenizer.save_pretrained(out_path)


