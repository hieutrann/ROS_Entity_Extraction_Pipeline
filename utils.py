import pprint as p

"""Functions Required for ROS Segmentation"""
#====================================================================
import pandas as pd
def sectag_to_regex(header_file_path, seg_col, header_col):
  header_df = pd.read_csv(header_file_path)
  header_df = header_df.drop_duplicates()
  headers = header_df[header_col].tolist()
  header_patterns = [f'^{header}[\n:]' for header in headers]
  return header_patterns, header_df[seg_col].tolist()

#====================================================================
import re
def find_segs(note, header_patterns, seg_names):
  segs = {}

  # Find the section headers and their start positions
  for i, pattern in enumerate(header_patterns):
    for m in re.finditer(pattern, note.lower(), re.MULTILINE):
      seg_head = (note[m.span()[0]:m.span()[1]], m.span()[0])
      if seg_head not in segs:
        segs[seg_head] = []  # A seg head can have multiple general seg names

      segs[seg_head].append(seg_names[i]) 

  segs = [[k[0], segs[k], k[1]] for k in segs.keys()]
  segs = sorted(segs, key=lambda x: x[2])
  
  # Find the entir sections and their start and end positions
  for i in range(len(segs)):
    if i == len(segs)-1:
      segs[i].append(len(note))
    else:
      segs[i].append(segs[i+1][2])

  return segs

#====================================================================
def ros_seg(note, segs):
  ros_data = []   # ros text + start position
  right_after_ros = False # flag sections after ROS
  for seg in segs:
      section_names = seg[1]
      section_content = note[seg[2]:seg[3]]  

      #---------------------------------------------------------------
      # Handle hiearchical subsections within ROS if exist
      if right_after_ros:
        if any("review" in item for item in section_names):
          # If there is a review section, append it to the ros_data
          ros_data[0] += section_content
        else:
          right_after_ros = False  # Consider it goes to another section
     
      #---------------------------------------------------------------
      if 'review_of_systems' in section_names:
          ros_data = [section_content, seg[2]]
          right_after_ros = True

  return ros_data

"""Other functions"""
#Remove special characters from json
def remove_char_json(text):
  pattern = r'\[.*\]'
  match = re.search(pattern, text, re.DOTALL)
  result = match.group()
  return result

#Clean up, organize ros entities captured from LLMs output: 
###remove unnecessary characters and group them into ros category
def regexp_ros(text):
  pattern = r"-->\s*(?P<ros>[^(\n]+)"
  m = re.search(pattern, text, re.I)
  if m:
    return m.group('ros').strip()
  else:
    return 'None'

#Return maximum similar text - using sentence transformer
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def return_max_sim_text(pred_tokens_encode, original_tokens, original_tokens_seq):
  n = len(original_tokens)
  # max_sim_text = ''
  max_sim_text_seq = []; max_sim = 0

  # for w in range(1, n):
  for w in range(1, n):
    i = 0
    while i+w <= n:
      #join possible tokens from original text, starting from 0:1
      tokens_comparing = ' '.join(original_tokens[i:i+w])
      tokens_comparing_encode = model.encode(tokens_comparing)  #encoding

      #calculate the similarity, keep the highest similarity
      sim = util.cos_sim(tokens_comparing_encode, pred_tokens_encode).item()
      if sim > max_sim:
        max_sim = sim
        max_sim_text_seq = [original_tokens_seq[i] for i in range(i,i+w)]
      i += 1
    
    """ USE THIS FOR OPTIMAL EXECUTION TIME AND PERFORMANCE MEASURING
    start_time = time.perf_counter()

    spans_dup = []
    for i in range(0, n):
        for w in range(1, n+1):
            spans_dup.append(' '.join(clean_original_tokens[i:i+w]))
    spans = list(set(spans_dup)) #Deduplication

    #Embedding
    spans_emb = model.encode(spans)

    alg_pred_tokens = []

    for token in pred_tokens:    
        # Encode target
        target_emb = model.encode(token)
        sim = util.cos_sim(target_emb, spans_emb) 

        #Mapping back all values = highest value
        top_idx = sim.argmax()

        #Map back embedded vector to tokens
        best_match = spans[top_idx]
        alg_pred_tokens.append(best_match)

    # Record the end time
    end_time = time.perf_counter() """

  return max_sim_text_seq

def return_text(pred_tokens, original_tokens, original_tokens_seq):
  n = len(original_tokens)
  # max_sim_text = ''
  max_sim_text_seq = []

  # for w in range(1, n):
  for w in range(1, n):
    i = 0
    while i+w <= n:
      #join possible tokens from original text, starting from 0:1
      tokens_comparing = ' '.join(original_tokens[i:i+w])
      if pred_tokens == tokens_comparing:         
        max_sim_text_seq = [original_tokens_seq[i] for i in range(i,i+w)]
      # else:
      #    max_sim_text_seq.append({'unknown_entity':'over_detection'})
        # max_sim_text = tokens_comparing
        # print(max_sim_text_seq)
      i += 1
    
  return max_sim_text_seq



#drop the status pair (last pair) in a dictionary
def drop_last_pair(dicts):
    result = []
    for d in dicts:
        if d:
            new_d = dict(list(d.items())[:-1])
            result.append(new_d)
        else:
            result.append(d)
    return result

#Reappend the status key-value pair by comparing matched_preds and matched_labels with pre-trim list, return original
def return_original_dict(pre_trim, matched):
    original_dict = []
    for b in matched:
        b_keys = set(b.keys())
        for a in pre_trim:
            if b_keys.issubset(a.keys()): #all keys of b are in a
                original_dict.append(a)
                break
    return original_dict

## Extract label and prediction for EM, RM, OD, UD
def extract_labels_preds(label_list, pred_list):
    exact_matched_labels = []
    exact_matched_preds = []

    relax_matched_labels = []
    relax_matched_preds = []  

    over_detection = []
    under_detection = []

    label_list_trim = drop_last_pair(label_list)
    pred_list_trim = drop_last_pair(pred_list)

    for label in label_list_trim:
        for pred in pred_list_trim:
            # Find overlapping keys
            # print(set(pred.keys()).difference('status'))
            common_keys = set(pred.keys()) & set(label.keys())

            if common_keys:
                #Check if it's exact match (all keys match)
                if set(pred.keys()) == set(label.keys()):
                    exact_matched_labels.append(label)
                    exact_matched_preds.append(pred)
                else: #relax match
                    relax_matched_labels.append(label)
                    relax_matched_preds.append(pred)

    #over/under detection
    for pred in pred_list_trim:
        if pred not in exact_matched_labels and pred not in relax_matched_labels and pred not in exact_matched_preds and pred not in relax_matched_preds:
                # print(exact_matched_labels)
                # print(relax_matched_labels)
                # print(relax_matched_preds)
                # print(pred)
                over_detection.append(pred)

    # print(over_detection)         
    for label in label_list_trim:
        if label not in exact_matched_preds and label not in relax_matched_preds and label not in exact_matched_labels and label not in relax_matched_labels:
                under_detection.append(label)
                
    #mapping back with the original
    exact_matched_labels_final = return_original_dict(label_list, exact_matched_labels)
    exact_matched_preds_final = return_original_dict(pred_list, exact_matched_preds)
    relax_matched_labels_final = return_original_dict(label_list, relax_matched_labels)
    relax_matched_preds_final = return_original_dict(pred_list, relax_matched_preds)

    over_detection_final = return_original_dict(pred_list, over_detection)
    under_detection_final = return_original_dict(label_list, under_detection)

    print(f"OD : {over_detection_final}")
    # return matched_labels_final, matched_preds_final
    return exact_matched_labels_final, exact_matched_preds_final, relax_matched_labels_final, relax_matched_preds_final, over_detection_final, under_detection_final

## EXACT MATCH performance
def exact_match_metrics(em_labels, em_preds, ud, rm_preds, od):
  list_tp = []
  list_fp = []

  for d1, d2 in zip(em_labels, em_preds):
      if d1 == d2:
          list_tp.append(d1)
      else:
          list_fp.append({"label":d1, "pred":d2})

  tp = len(list_tp)
  fp = len(list_fp)
  p.pprint(list_fp)
  ud = len(ud)
  od = len(rm_preds) + len(od)

  if (tp+fp+od) == 0:
     precision_em = 0
  
  elif (tp+ud) == 0:
     sensitivity_em = 0
  else:
    precision_em = round((tp / (tp+fp+od)),2)
    sensitivity_em = round((tp / (tp+ud)),2)

  if precision_em + sensitivity_em == 0:
    f1_em = 0
  else:
      f1_em = round((2 * precision_em * sensitivity_em / (precision_em + sensitivity_em)),2)
  print(f"em:{tp}, fp:{fp}, ud:{ud},od: {od}")
  print(precision_em, sensitivity_em, f1_em)
  return precision_em, sensitivity_em, f1_em, tp, fp, ud, od

## RELAX MATCH performance
def relax_match_metrics(em_labels, em_preds, ud, rm_labels, rm_preds, od):
  list_tp = []
  list_fp = []

  #Including the exact match
  for d1, d2 in zip(em_labels, em_preds):
      if d1 == d2:
          list_tp.append(d1)
      else:
          list_fp.append({"label":d1, "pred":d2})

  #Including the relax match
  for d1, d2 in zip(rm_labels, rm_preds):
      if d1['status'] == d2['status']:
          list_tp.append(d1)
      else:
          list_fp.append({"label":d1, "pred":d2})

  tp = len(list_tp)
  fp = len(list_fp)
  p.pprint(list_fp)
  ud = len(ud)
  od = len(od)

  if (tp+fp+od) == 0:
     precision_rm = 0
  
  elif (tp+ud) == 0:
     sensitivity_rm = 0
  else:
    precision_rm = round((tp / (tp+fp+od)),2)
    sensitivity_rm = round((tp / (tp+ud)),2)

  if precision_rm + sensitivity_rm == 0:
    f1_rm = 0
  else:
      f1_rm = round((2 * precision_rm * sensitivity_rm / (precision_rm + sensitivity_rm)),2)
  print(f"rm:{tp}, fp:{fp}, ud:{ud},od: {od}")
  print(precision_rm, sensitivity_rm, f1_rm)
  return precision_rm, sensitivity_rm, f1_rm, tp, fp, ud, od

