import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import PIL
import torch
import torch.nn.functional as F
import select
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import fastprogress
import sys
from fastai.callback.all import *
from fastcore.utils import *
from fastcore.foundation import *
sys.path.insert(0, '/home/jupyter/VLP/')
from pytorch_pretrained_bert.tokenization import BertTokenizer

HATE_PATH = Path('/home/jupyter/mmf_data/datasets/hateful_memes/defaults/')
HATE_IMAGES = HATE_PATH/'images'
HATE_ANNOT = HATE_PATH/'annotations'
HATE_FEAT_PATH = Path('/home/jupyter/hateful_features/region_feat_gvd_wo_bgd')
PHASE_2 = Path('/home/jupyter/hate_phase2')

class VLPInput(tuple):pass

def gen_submit(learn, fname, softmax=False, test_file = 'test_seen.jsonl'):
    test_df = pd.read_json(PHASE_2/test_file, lines=True)
    test_dl = learn.dls.test_dl(test_df)
    preds = learn.get_preds(dl=test_dl)[0]
    if softmax:
        preds = F.softmax(preds, dim=1)
    _, indcs = preds.max(dim=1)
    probs= preds[:,1]
    submit_df = pd.DataFrame()
    submit_df['id'] = test_df['id']
    submit_df['proba'] = probs
    submit_df['label'] = indcs
    submit_df = submit_df.set_index('id')
    submit_df.to_csv(fname, header=True)
    
    
def id_to_img_path(id):
    id = f'{int(id):05d}'
    return str(PHASE_2/f'img/{id}.png')

def id_to_text(id, data):
    return first(data[data.id==id].text)

def id_to_label(id, data):
    return first(data[data.id==id].label)

def load_from_row(row, proc, tokenizer, q=None):
    img_id = row.id
    text = row.text
    if q is None: q = text
    img_file = id_to_img_path(img_id)
    instance = (img_file, tokenizer.tokenize(q), {'answers': ['dummy']})
    input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next, task_idx, conv_feats, vis_masked_pos, vis_pe, ans_labels= proc(instance)
    return VLPInput((torch.tensor(img_id), conv_feats, vis_pe, input_ids, segment_ids, input_mask))  


def find_like(s, data):
    return data[data.text.str.contains(s, case=False)]

def by_ids(ids, data):
    return data[data.id.isin(ids)]

def show(data):
    n = min(len(data), 40)
    _,axs = plt.subplots((n+1)//2,2, figsize=(20,2*n))
    for ax, (_,row) in zip(axs.flatten(), data.iterrows()):
        img_path = id_to_img_path(row['id'])
        ax.imshow(PIL.Image.open(img_path))
        ax.axis('off')
        clr = 'red' if row['label']==1 else 'green'
        txt = f'{row["id"]}: {row["tex_cap"][:80]}'
        ax.set_title(txt, color=clr) 
        
        import select

common_words = [
    (['muslim'], ''),
    (['omar', 'ilhan', 'mohammed', 'ahmed', 'tlaib', 'mohammad', 'rashida'], 'muslim name'),
    (['trump', 'donald'], 'donald trump'),
    (['islam', 'allah', 'sharia'], 'islam'),
    (['jew'], ''),
    (['anne','frank'], 'jew girl name'),
    (['america', 'usa'], 'america'),
    (['tranny'], ''),
    (['obama'], 'black name'),
    (['michelle', 'barack'], 'obama black name'),
    (['hitler'],'german name killer'),
    (['isis'], 'muslim terrorist'),
    (['polish'],''),
    (['democrat'],''),
    (['liberal'], ''),
    (['republican'], ''),
    (['christian'], ''),
    (['nazi'],'german killer'),
    (['meme'],''),
    (['mexican'],''),
    (['jesus'],''),
    (['africa',],''),
    (['porn'],''),
    (['walmart'],''),
    (['retarded'],'stupid curse'),
    (['auschwitz', 'holocaust'],'jew death'),
    (['pussy'],''),
    (['chinese', 'asian'], 'asia'),
    (['hillary', 'clinton'],'woman name'),
    (['jenner'],'trans name'),
    (['gorilla', 'apes'],'monkey'),
    (['christchurch'],'muslim death'),
    (['hijab'],'muslim wear'),
    (['hump'],'fuck'),
    (['kfc'],'restaurant'),
    (['pic'], ''),
    (['diaper'],''),
    (['lol'],''),
    (['bro'],''),
    (['nigger'],'black curse'),
    (['tyrone'],'black name'),
    (['israel'], 'jew country'),
    ]
def get_tokenizer(with_replace=False):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    if with_replace:
        cur_idx = 12
        for awords, bwords in common_words:
            for word in awords:
                tokenizer.vocab[word] = cur_idx 
                cur_idx +=1
    return tokenizer