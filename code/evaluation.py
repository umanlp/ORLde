
def orl_evaluation(gold_original, system_original, dict_source, dict_target):
    system_source = False
    system_target = False
    system_speaker = False
    system_source_list = []
    system_target_list = []
    
    gold_source = False
    gold_target = False
    gold_speaker = False
    gold_source_list = []
    gold_target_list = []
    
    source_tp = 0
    source_fp = 0
    source_fn = 0
    target_tp = 0
    target_fp = 0
    target_fn = 0

    system = system_original
    gold = gold_original
    
    i = 0
    while i < len(gold_original):
        gold[i] = gold_original[i].strip("B-")
        gold[i] = gold_original[i].strip("I-")
        if gold[i] == "Source":
            gold_source = True
            gold_source_list.append(i)
        elif gold[i] == "Target":
            gold_target = True
            gold_target_list.append(i)
        elif gold[i] == "Sprecher":
            gold_speaker = True

        i = i+1

    j = 0
    while j < len(system_original):
        system[j] = system_original[j].strip("B-")
        system[j] = system_original[j].strip("I-")
        if system[j] == "Source":
            system_source = True
            system_source_list.append(j)
        elif system[j] == "Target":
            system_target = True
            system_target_list.append(j)
        elif system[j] == "Sprecher":
            system_speaker = True

        j = j+1
    
    system_source_list.sort()
    system_target_list.sort()
    gold_source_list.sort()
    gold_target_list.sort()
    
    #print("Gold source: ", gold_source_list)
    #print("Gold target: ", gold_target_list)
    #print("System source: ", system_source_list)
    #print("System target: ", system_target_list)
    #print("\n")
    
    
    
    #### Case 1a: Gold source is empty####
    if not gold_source and not gold_speaker:
        if not system_source and not system_speaker:
            pass
        elif system_source or system_speaker:
            source_tp = 0
            source_fp = 1
            source_fn = 0


    #### Case 1b: Gold target is empty ####
    if not gold_target:
        if not system_target:
            pass
        else:
            target_tp = 0
            target_fp = 1
            target_fn = 0
        
    #### Case 2: Gold source is only speaker ####
    if not gold_source and gold_speaker:
        if not system_source and not system_speaker:
            source_tp = 0
            source_fp = 0
            source_fn = 1
        elif system_source and not system_speaker:
            source_tp = 0
            source_fp = 1
            source_fn = 1
        elif system_speaker:
            source_tp = 1
            source_fp = 0
            source_fn = 0
    
    ### Case 3: Gold source is speaker and terminal ###
    if gold_source and gold_speaker:
        if system_source and not system_speaker: 
            if system_source_list == gold_source_list: 
                source_tp = 1
                source_fp = 0
                source_fn = 0
            else: 
                source_tp = 0
                source_fp = 1
                source_fn = 1
        if system_source and system_speaker: #11
            overlap = set(set(system_source_list).intersection(gold_source_list))
            if system_source_list == gold_source_list: 
                source_tp = 1
                source_fp = 0
                source_fn = 0
            elif overlap:
                source_tp = 0
                source_fp = 1
                source_fn = 1
            else: #no overlap
                source_tp = 1
                source_fp = 0
                source_fn = 0
        if not system_source and system_speaker: 
            source_tp = 1 
            source_fp = 0
            source_fn = 0
        if not system_source and not system_speaker: 
            source_tp = 0 
            source_fp = 0
            source_fn = 1
    
    
    #### Case 4a: gold source is only terminal ####
    if gold_source and not gold_speaker:
        if not system_source and not system_speaker:
            source_tp = 0
            source_fp = 0
            source_fn = 1
        elif system_source and not system_speaker:
            if system_source_list == gold_source_list:
                source_tp = 1
                source_fp = 0
                source_fn = 0
            else:
                source_tp = 0
                source_fp = 1
                source_fn = 1
        elif not system_source and system_speaker:
            source_tp = 0
            source_fp = 1
            source_fn = 1
        elif system_source and system_speaker:
            if system_source_list == gold_source_list:
                source_tp = 1
                source_fp = 0
                source_fn = 0
            else:
                source_tp = 0
                source_fp = 1
                source_fn = 1
            
    ### Case 4b: gold target is terminal ####
    if gold_target:
        if not system_target:
            target_tp = 0
            target_fp = 0
            target_fn = 1
        else:
            if system_target_list == gold_target_list:
                target_tp = 1
                target_fp = 0
                target_fn = 0
            else:
                target_tp = 0
                target_fp = 1
                target_fn = 1
    
    
    dict_source["source_tp"] = dict_source["source_tp"] + source_tp
    dict_source["source_fp"] = dict_source["source_fp"] + source_fp
    dict_source["source_fn"] = dict_source["source_fn"] + source_fn
    dict_target["target_tp"] = dict_target["target_tp"] + target_tp
    dict_target["target_fp"] = dict_target["target_fp"] + target_fp
    dict_target["target_fn"] = dict_target["target_fn"] + target_fn
    
    
    return source_tp, source_fp, source_fn, target_tp, target_fp, target_fn, dict_source, dict_target


# Calculates the micro average of recall, precision and F1 for source and target predictions

def orl_micro_average(dict_source, dict_target):
    source_tp = dict_source["source_tp"]
    source_fp = dict_source["source_fp"]
    source_fn = dict_source["source_fn"]
    source_recall = 0
    source_precision = 0
    source_f1 = 0
    
    target_tp = dict_target["target_tp"] 
    target_fp = dict_target["target_fp"] 
    target_fn = dict_target["target_fn"] 
    target_recall = 0
    target_precision = 0
    target_f1 = 0
    
    
    if source_tp > 0: 
        source_recall = source_tp/(source_tp+source_fn)
        source_precision = source_tp/(source_tp+source_fp)
        if (source_recall + source_precision) > 0:
          source_f1 = (2*source_recall*source_precision)/(source_recall+source_precision)
    
    if target_tp > 0:
        target_recall = target_tp/(target_tp + target_fn)
        target_precision = target_tp/(target_tp + target_fp)
        if (target_recall + target_precision) > 0: 
          target_f1 = (2*target_recall*target_precision)/(target_recall+target_precision)
    
    print("ORL Results:")
    print("Source:")
    print("TP: {}, FP: {}, FN: {}".format(source_tp, source_fp, source_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(source_recall, 3), round(source_precision, 3), round(source_f1, 3)))
    print("")
    print("Target:")
    print("TP: {}, FP: {}, FN: {}".format(target_tp, target_fp, target_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(target_recall, 3), round(target_precision, 3), round(target_f1, 3)))

    return source_recall, source_precision, source_f1, target_recall, target_precision, target_f1

# Evaluation on token level
# No distinction between the individual roles

def srl_micro_average(dict_srl):
    srl_tp = dict_srl["srl_tp"]
    srl_fp = dict_srl["srl_fp"]
    srl_fn = dict_srl["srl_fn"]
    srl_recall = 0
    srl_precision = 0
    srl_f1 = 0

    if srl_tp > 0:
        srl_recall = srl_tp/(srl_tp+srl_fn)
        srl_precision = srl_tp/(srl_tp+srl_fp)
        if (srl_recall + srl_precision) > 0:
          srl_f1 = (2*srl_recall*srl_precision)/(srl_recall+srl_precision)

    print("SRL Results:")
    print("TP: {}, FP: {}, FN: {}".format(srl_tp, srl_fp, srl_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(srl_recall, 3), round(srl_precision, 3), round(srl_f1, 3)))
    print("\n")

    return srl_recall, srl_precision, srl_f1



def dep_evaluation(gold_original, system_original, dict_dep):
    dep_tp = 0
    dep_fp = 0
    dep_fn = 0
    system = system_original
    gold = gold_original

    j = 0
    while j < len(system_original):
      system[j] = system_original[j].strip("B-")
      system[j] = system_original[j].strip("I-")
      j = j+1


    i = 0
    while i < len(gold_original):
        gold[i] = gold_original[i].strip("B-")
        gold[i] = gold_original[i].strip("I-")


        if i < len(system):
          if gold[i] != "O" and gold[i] != "X":
            if gold[i] == system[i]:
              dep_tp = dep_tp + 1
            elif system[i] == "O":
                dep_fn = dep_fn +1
            else:
                dep_fn = dep_fn +1
                dep_fp = dep_fp +1

          else:
            if gold[i] == system[i]: # both "O" or "X"
              pass
            else:
             dep_fp = dep_fp + 1

        i = i+1

    dict_dep["dep_tp"] = dict_dep["dep_tp"] + dep_tp
    dict_dep["dep_fp"] = dict_dep["dep_fp"] + dep_fp
    dict_dep["dep_fn"] = dict_dep["dep_fn"] + dep_fn

    return dep_tp, dep_fp, dep_fn, dict_dep



def dep_micro_average(dict_dep):
    dep_tp = dict_dep["dep_tp"]
    dep_fp = dict_dep["dep_fp"]
    dep_fn = dict_dep["dep_fn"]
    dep_recall = 0
    dep_precision = 0
    dep_f1 = 0

    if dep_tp > 0:
        dep_recall = dep_tp/(dep_tp+dep_fn)
        dep_precision = dep_tp/(dep_tp+dep_fp)
        if (dep_recall + dep_precision) > 0:
          dep_f1 = (2*dep_recall*dep_precision)/(dep_recall+dep_precision)

    print("DEP Results:")
    print("TP: {}, FP: {}, FN: {}".format(dep_tp, dep_fp, dep_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(dep_recall, 3), round(dep_precision, 3), round(dep_f1, 3)))
    print("\n")

    return dep_recall, dep_precision, dep_f1





# Evaluation on token level
# No distinction between the individual roles

def srl_evaluation(gold_original, system_original, dict_srl):

    srl_tp = 0
    srl_fp = 0
    srl_fn = 0
    system = system_original
    gold = gold_original
    
    j = 0
    while j < len(system_original):
      system[j] = system_original[j].strip("B-")
      system[j] = system_original[j].strip("I-")
      j = j+1


    i = 0
    while i < len(gold_original):
        gold[i] = gold_original[i].strip("B-")
        gold[i] = gold_original[i].strip("I-")


        if i < len(system):
          if gold[i] != "O" and gold[i] != "X":
            if gold[i] == system[i]:
              srl_tp = srl_tp + 1
            elif system[i] == "O":
                srl_fn = srl_fn +1
            else:
                srl_fn = srl_fn +1
                srl_fp = srl_fp +1
                
          else:
            if gold[i] == system[i]: # both "O" or "X"
              pass
            else:
             srl_fp = srl_fp + 1
       
        i = i+1
        
    dict_srl["srl_tp"] = dict_srl["srl_tp"] + srl_tp
    dict_srl["srl_fp"] = dict_srl["srl_fp"] + srl_fp
    dict_srl["srl_fn"] = dict_srl["srl_fn"] + srl_fn
                
    return srl_tp, srl_fp, srl_fn, dict_srl



def srl_micro_average(dict_srl):
    srl_tp = dict_srl["srl_tp"]
    srl_fp = dict_srl["srl_fp"]
    srl_fn = dict_srl["srl_fn"]
    srl_recall = 0
    srl_precision = 0
    srl_f1 = 0
    
    if srl_tp > 0: 
        srl_recall = srl_tp/(srl_tp+srl_fn)
        srl_precision = srl_tp/(srl_tp+srl_fp)
        if (srl_recall + srl_precision) > 0:
          srl_f1 = (2*srl_recall*srl_precision)/(srl_recall+srl_precision)
    
    print("SRL Results:")
    print("TP: {}, FP: {}, FN: {}".format(srl_tp, srl_fp, srl_fn))
    print("Recall: {}, Precision: {}, F1: {}".format(round(srl_recall, 3), round(srl_precision, 3), round(srl_f1, 3)))
    print("\n")

    return srl_recall, srl_precision, srl_f1



