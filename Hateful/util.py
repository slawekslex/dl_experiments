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
from fastai.callback.all import *
from fastcore.utils import *

HATE_PATH = Path('/home/jupyter/mmf_data/datasets/hateful_memes/defaults/')
HATE_IMAGES = HATE_PATH/'images'
HATE_ANNOT = HATE_PATH/'annotations'
HATE_FEAT_PATH = Path('/home/jupyter/hateful_features/region_feat_gvd_wo_bgd')

class VLPInput(tuple):pass

def gen_submit(learn, fname, softmax=False):
    test_df = pd.read_json(HATE_ANNOT/'test.jsonl', lines=True)
    test_dl = learn.dls.test_dl(test_df)
    preds = learn.get_preds(dl=test_dl)[0]
    if softmax:
        preds = F.softmax(preds)
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
    return str(HATE_PATH/f'images/img/{id}.png')

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
        img_path = HATE_IMAGES / row['img']
        ax.imshow(PIL.Image.open(img_path))
        ax.axis('off')
        clr = 'red' if row['label']==1 else 'green'
        txt = f'{row["id"]}: {row["tex_cap"][:80]}'
        ax.set_title(txt, color=clr) 
        
        import select


