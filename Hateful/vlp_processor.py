from random import randint, shuffle, choices
from random import random as rand
import h5py
from vlp.loader_utils import get_random_word, Pipeline
from vlp.seq2seq_loader import truncate_tokens_pair
import torch
import numpy as np
import torch.nn.functional as F
class PreprocessVLP(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, block_mask=False, truncate_config={}, mask_image_regions=False, mode="s2s", len_vis_input=49, vis_mask_prob=0.25, 
                  region_bbox_prefix='',  region_bbox_file = None, region_det_file_prefix='', local_rank=-1, load_vqa_ann=False, id_digits=3):
        super().__init__()
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.mask_image_regions = mask_image_regions
        assert mode in ("s2s", "bi")
        self.mode = mode
        self.region_bbox_prefix = region_bbox_prefix
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix
        self.id_digits = id_digits


        self.len_vis_input = len_vis_input
        self.vis_mask_prob = vis_mask_prob
        self.task_idx = 0
        # for images
        if load_vqa_ann:
            # import packages from pythia
            import pythia.tasks.processors as pythia_proc # VQAAnswerProcessor
            from pythia.utils.configuration import ConfigNode
            args = {'vocab_file': '/home/jupyter/VLP/pythia/data/vocabs/answers_vqa.txt', 'num_answers':10, 'preprocessor':{'type':'simple_word', 'params':{}}}
            args = ConfigNode(args)
            self.ans_proc = pythia_proc.registry.get_processor_class('vqa_answer')(args)
        else:
            self.ans_proc = None


    def __call__(self, instance):
        img_path, tokens_b = instance[:2]
        tokens_a = ['[UNK]'] * self.len_vis_input

        truncate_tokens_pair(tokens_a, tokens_b,
            self.len_vis_input + self.max_len_b, max_len_b=self.max_len_b,
            trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)
       

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length * self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = cand_pos[:n_pred]

        if self.mask_image_regions:
            vis_masked_pos = np.random.choice(self.len_vis_input,
                int(self.len_vis_input*self.vis_mask_prob), replace=False)+1 # +1 for [CLS], always of the same length, no need to pad
        else:
            vis_masked_pos = []

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        second_st, second_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3

        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            input_mask = torch.tensor([1] * len(tokens) + [0] * n_pad, dtype=torch.long) \
                .unsqueeze(0).expand(self.max_len, self.max_len).clone()

        #import pdb;pdb.set_trace()
        if self.mask_image_regions:
            #input_mask[:, vis_masked_pos].fill_(0) # block the masked visual feature
            input_mask[:, vis_masked_pos] = 0 # block the masked visual feature

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        # loading pre-processed features
        img_id = img_path.split('/')[-1].split('.')[0]
        if self.region_bbox_file is not None:
            bbox_file = self.region_bbox_file
        else:
            bbox_file = self.region_bbox_prefix+'_bbox'+img_id[-self.id_digits:]+'.h5'
        if self.region_det_file_prefix != '':
            # read data from h5 files
            with h5py.File(self.region_det_file_prefix+'_feat'+img_id[-self.id_digits:] +'.h5', 'r') as region_feat_f, \
                    h5py.File(self.region_det_file_prefix+'_cls'+img_id[-self.id_digits:] +'.h5', 'r') as region_cls_f, \
                    h5py.File(bbox_file, 'r') as region_bbox_f:
                img = torch.from_numpy(region_feat_f[img_id][:]).float()
                cls_label = torch.from_numpy(region_cls_f[img_id][:]).float()
                vis_pe = torch.from_numpy(region_bbox_f[img_id][:])
        else:
            # legacy, for some datasets, read data from numpy files
            img = torch.from_numpy(np.load(img_path))
            cls_label = torch.from_numpy(np.load(img_path.replace('.npy', '_cls_prob.npy')))
            with h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                vis_pe = torch.from_numpy(region_bbox_f[img_id][:])

        # lazy normalization of the coordinates...
        w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
        h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
        vis_pe[:, [0, 2]] /= w_est
        vis_pe[:, [1, 3]] /= h_est
        assert h_est > 0, 'should greater than 0! {}'.format(h_est)
        assert w_est > 0, 'should greater than 0! {}'.format(w_est)
        rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
        rel_area.clamp_(0)

        vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1) # confident score
        normalized_coord = F.normalize(vis_pe.data[:, :5]-0.5, dim=-1)
        vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
            F.layer_norm(cls_label, [1601])), dim=-1) # 1601 hard coded...

        
        ans_tk = img.new(1)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, -1, self.task_idx, img, vis_masked_pos, vis_pe, ans_tk)
