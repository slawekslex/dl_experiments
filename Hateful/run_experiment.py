import numpy as np
import torch
import random
import pickle
import sys
import os
import glob
import pdb
import argparse

sys.path.insert(0, '/home/jupyter/VLP/pythia')
sys.path.insert(0, '/home/jupyter/VLP/')

from torch.distributions.beta import Beta
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam
from pathlib import Path
import pandas as pd
from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
import PIL
from vlp.lang_utils import language_eval

from fastai.vision.all import *

from vlp_processor import PreprocessVLP
import pythia.tasks.processors as pythia_proc

from util import *


class HateStem(torch.nn.Module):
    def __init__(self, vlp):
        super(HateStem, self).__init__()
        self.add_module('vis_embed',vlp.vis_embed) #Linear->ReLU->Linear->ReLU->dropout
        self.vis_pe_embed = vlp.vis_pe_embed #Linear->ReLU->dropout
        self.bert = vlp.bert # pytorch_pretrained_bert.modeling.BertModel
        self.len_vis_input = vlp.len_vis_input
        
    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids=None, attention_mask=None):
        vis_feats = self.vis_embed(vis_feats) # image region features
        vis_pe = self.vis_pe_embed(vis_pe) # image region positional encodings

        sequence_output, pooled_output = self.bert(vis_feats, vis_pe, input_ids, token_type_ids,
            attention_mask, output_all_encoded_layers=False, len_vis_input=self.len_vis_input)
        #print(sequence_output.shape, pooled_output.shape)
        vqa2_embed = sequence_output[:, 0]*sequence_output[:, self.len_vis_input+1]
        return vqa2_embed
        #return sequence_output
        
class HateClassifier(torch.nn.Module):
    def __init__(self, stem):
        super(HateClassifier, self).__init__()
        self.stem = stem
        self.classifier = create_head(768,2, lin_ftrs=[args.head_linear], ps=args.head_ps)
    def forward(self, params, lam=None, shuffle=None):  
        id, vis_feats, vis_pe, input_ids, token_type_ids, attention_mask = params
        embs = self.stem(vis_feats, vis_pe, input_ids, token_type_ids, attention_mask)
        if lam is not None:
            embs2 = embs[shuffle]
            embs = torch.lerp(embs2, embs, lam)
        return self.classifier(embs)
        
def create_head(nf, n_out, lin_ftrs=None, ps=0.5, bn_final=False, lin_first=False, ):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    ps = L(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = []
    layers = [Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)
    
def new_model():
    hate_stem = torch.load(args.stem_file)
    stem_modules = list(hate_stem.modules())
    dropouts = [x for x in stem_modules if isinstance(x, torch.nn.modules.dropout.Dropout)]
    for x in dropouts: x.p = args.stem_ps
    return  HateClassifier(hate_stem).cuda()

def get_y(row):
    return int(row['label'])

class LoadRow(Transform):
    def __init__(self, split_idx, processor, tokenizer):
        self.split_idx = split_idx
        self.proc = processor
        self.tokenizer = tokenizer
    
    def encodes(self, x, **kwargs):
        return load_from_row(x, self.proc, self.tokenizer)
    
def vlp_splitter(model):
    return L(params(model.stem.vis_embed) + params(model.stem.vis_pe_embed), 
            params(model.stem.bert),
            params(model.classifier))

class MixUp(Callback):
    run_after,run_valid = [Normalize],False
    def __init__(self, alpha=0.4): self.distrib = Beta(tensor(alpha), tensor(alpha))
    def before_fit(self):
        self.stack_y = getattr(self.learn.loss_func, 'y_int', False)
        if self.stack_y: self.old_lf,self.learn.loss_func = self.learn.loss_func,self.lf

    def after_fit(self):
        if self.stack_y: self.learn.loss_func = self.old_lf

    def before_batch(self):        
        lam = self.distrib.sample((self.y.size(0),)).squeeze().cuda()
        lam = torch.stack([lam, 1-lam], 1)
        self.lam = lam.max(1)[0]
        shuffle = torch.randperm(self.y.size(0)).cuda()
        xb, yb = self.learn.xb[0], self.learn.yb[0]
        self.yb1 = (yb[shuffle],)
        self.learn.xb =self.xb+ (self.lam.unsqueeze(1), shuffle)
        

    def lf(self, pred, *yb):
        if not self.training: return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            loss = torch.lerp(lf(pred,*self.yb1), lf(pred,*yb), self.lam)
        return reduce_loss(loss, getattr(self.old_lf, 'reduction', 'mean'))
    
def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--experiment_id")
    parser.add_argument("--max_tgt_length", default=100, type=int, help="maximum length text tokens")
    parser.add_argument("--head_ps", default=.5, type=float, help='dropout in head')
    parser.add_argument("--stem_ps", default=.2, type=float, help='dropout in stem')
    parser.add_argument("--head_linear", default=512, type=int, help='size of the middle linear layer')
    parser.add_argument("--max_masked", default=10, type=int, help='max masked text tokens')
    parser.add_argument("--mask_prob", default=.2, type=float, help='token mask prob')
    parser.add_argument("--vis_mask_prob", default=.2, type=float, help='img mask prob')
    parser.add_argument("--lr", default=0.001, type=float, help='initial learning rate for head')
    parser.add_argument("--lr_mult", default=10, type=float, help='difference in lr per split')
    parser.add_argument("--train_epochs", default=8, type=int, help='number of training epochs')
    parser.add_argument("--mixup_alpha", default=0.4, type=int, help='number of training epochs')
    parser.add_argument("--stem_file", type=str, help='pretreined model path')
    
    global args
    args = parser.parse_args()
    print(args)
    
    max_seq_length = args.max_tgt_length + 100 + 3
    
    # fix random seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    tokenizer.max_len = max_seq_length
    
    PHASE_2 = Path('/home/jupyter/hate_phase2')
    train = pd.read_json(PHASE_2/'train.jsonl', lines=True)#[:160]
    dev_seen = pd.read_json(PHASE_2/'dev_seen.jsonl', lines=True)#[:64]
    dev_unseen = pd.read_json(PHASE_2/'dev_unseen.jsonl', lines=True)#[:64]
    test_seen = pd.read_json(PHASE_2/'test_seen.jsonl', lines=True)
    test_unseen = pd.read_json(PHASE_2/'test_unseen.jsonl', lines=True)
    train['is_valid'] = False
    dev_seen['is_valid'] = True
    dev_unseen['is_valid']=True
    data = pd.concat([train, dev_seen, dev_unseen])
    
    
    region_pref = HATE_FEAT_PATH / 'feat_cls_1000/hateful_vlp_checkpoint_trainval'
    bbox_pref = HATE_FEAT_PATH / 'raw_bbox/hateful_vlp_checkpoint_trainval'
    id_digits=2

    truncate_config={
        'max_len_b': args.max_tgt_length, 'trunc_seg': 'b', 'always_truncate_tail': True}


    mask_img=True

    train_proc = PreprocessVLP(args.max_masked, args.mask_prob,
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, max_seq_length,
        truncate_config=truncate_config,mask_image_regions=mask_img, vis_mask_prob=args.vis_mask_prob,
        mode="bi", len_vis_input=100, 
        region_bbox_prefix=str(bbox_pref), region_det_file_prefix=str(region_pref), id_digits=id_digits,
        load_vqa_ann=True)

    val_proc = PreprocessVLP(0, 0,
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, max_seq_length,
        truncate_config=truncate_config,mask_image_regions=False, vis_mask_prob=0,
        mode="bi", len_vis_input=100, 
        region_bbox_prefix=str(bbox_pref), region_det_file_prefix=str(region_pref), id_digits=id_digits,
        load_vqa_ann=True)

    db = DataBlock(blocks = (TransformBlock, CategoryBlock), 
               get_x = Pipeline([LoadRow(0, train_proc, tokenizer),LoadRow(1, val_proc, tokenizer)]), 
               get_y=get_y,  splitter=ColSplitter('is_valid'))
    dls = db.dataloaders(data,bs=48)
    
    print(len(dls.train_ds), len(dls.valid_ds))
    roc_tracker = TrackerCallback(monitor='roc_auc_score', comp=np.greater)
    acc_tracker = TrackerCallback(monitor='accuracy', comp=np.greater)
    model = new_model()
    learn = Learner(dls, model,metrics=[accuracy, RocAucBinary()], cbs=[roc_tracker, acc_tracker, MixUp(args.mixup_alpha)], splitter=vlp_splitter)
    
    learn.fine_tune(args.train_epochs, args.lr, lr_mult=args.lr_mult)
    print('best:', roc_tracker.best, acc_tracker.best)
    
    exp_res = pd.read_csv('exp_results.csv')
    row = pd.DataFrame([{'id':args.experiment_id, 'hypers':str(args), 'roc':roc_tracker.best, 'acc':acc_tracker.best}])
    exp_res = exp_res.append(row)
    exp_res.to_csv('exp_results.csv', index=False)
if __name__ == '__main__':
    main()