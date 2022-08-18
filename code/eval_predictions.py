import ast
import json
import sys

"""
GOLD: ['O', 'O', 'O', 'B-Source', 'O', 'B-Target', 'I-Target', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
PRED: ['B-Source', 'I-Source', 'O', 'B-Source', 'O', 'B-Target', 'I-Target', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
"""
def eval_dic(dic):

    eval_dic = { "Source": { "tp":0, "fp":0, "fn":0 }, "Target": { "tp":0, "fp":0, "fn":0 }, "Speaker": { "tp":0, "fp":0, "fn":0 } }
    # sanity check
    if len(dic["gold"]) != len(dic["pred"]):
        print("LEN ERROR:", len(dic["gold"]), len(dic["pred"]))
        sys.exit()

    for i in range(len(dic["gold"])):
        tp, fp, fn = 0, 0, 0
        # remove prefix
        gold = [x.strip("B-").strip("I-") for x in dic["gold"][i]]
        pred = [x.strip("B-").strip("I-") for x in dic["pred"][i]]

        for j in range(len(gold)):
            if gold[j] != "O" and gold[j] != "X":
                if gold[j] == pred[j]:
                    eval_dic[gold[j]]['tp'] += 1
                elif pred[j] == "O":
                    eval_dic[gold[j]]['fn'] += 1
                else:
                    eval_dic[gold[j]]['fn'] += 1
                    eval_dic[pred[j]]['fp'] += 1
            else:
                if gold[j] == pred[j]: # both "O" or "X"
                    pass
                else:
                    eval_dic[pred[j]]['fp'] += 1

    return eval_dic


def orl_micro_average(dic, arg):
    orl_tp = dic[arg]["tp"]
    orl_fp = dic[arg]["fp"]
    orl_fn = dic[arg]["fn"]
    orl_recall = 0
    orl_precision = 0
    orl_f1 = 0

    if orl_tp > 0:
        orl_recall = orl_tp/(orl_tp+orl_fn)
        orl_precision = orl_tp/(orl_tp+orl_fp)
        if (orl_recall + orl_precision) > 0:
          orl_f1 = (2*orl_recall*orl_precision)/(orl_recall+orl_precision)

    print("ORL Results for", arg, ":")
    print("TP: {}, FP: {}, FN: {}".format(orl_tp, orl_fp, orl_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(orl_recall, 3), round(orl_precision, 3), round(orl_f1, 3)))
    print("\n")

    return orl_precision, orl_recall, orl_f1



def read_file(infile):
    sid = 1
    dic = { 'gold':[], 'pred':[] }
    data = []
    with open(infile, "r") as inf:
        for line in inf:
            if line.isspace():
                continue
            data.append(line)

    for i in range(len(data)):
        if data[i].strip() == str(sid):
            gold = data[i+2].strip().replace("GOLD: ", "")
            pred = data[i+3].strip().replace("PRED: ", "")
            dic['gold'].append(ast.literal_eval(gold))
            dic['pred'].append(ast.literal_eval(pred))
            sid += 1

    return dic


#####
# specify input logfile, e.g., logfile_ORL_BERT_run_1.log
infile = sys.argv[1]

dic = read_file(infile)
eval_dic = eval_dic(dic)

prec, rec, f1 = orl_micro_average(eval_dic, "Source")
print("SOURCE\tMICRO:\nprec:", prec, "rec:", rec, "F1:", f1)
prec, rec, f1 = orl_micro_average(eval_dic, "Target")
print("TARGET\tMICRO:\nprec:", prec, "rec:", rec, "F1:", f1)
prec, rec, f1 = orl_micro_average(eval_dic, "Speaker")
print("SPEAKER\tMICRO:\nprec:", prec, "rec:", rec, "F1:", f1)



