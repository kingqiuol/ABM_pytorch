# -*- coding: utf-8 -*-
from unicodedata import name
from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
MODULE_PATH = os.path.abspath(".")
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from model.densenet import *
from model.decoder import *


# create gru init state
class FcLayer(nn.Module):
    def __init__(self, nin, nout):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        out = torch.tanh(self.fc(x))
        return out


# Embedding
class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(config.K,config.M)

    def forward(self, y):
        emb = self.embedding(y)
        return emb
    
# calculate probabilities
class Gru_prob(nn.Module):
    def __init__(self, config):
        super(Gru_prob, self).__init__()
        self.fc_Wct = nn.Linear(config.D, config.M)
        self.fc_Wht = nn.Linear(config.N, config.M)
        self.fc_Wyt = nn.Linear(config.M, config.M)


        self.dropout = nn.Dropout(p=0.2)
        self.fc_W0 = nn.Linear(int(config.M / 2), config.K)

    def forward(self, cts, hts, emb, use_dropout):
        logit = self.fc_Wct(cts) + self.fc_Wht(hts) + self.fc_Wyt(emb)

        # maxout
        shape = logit.shape
        shape2 = int(shape[2] / 2)
        shape3 = 2
        logit = logit.view(shape[0], shape[1], shape2, shape3)
        logit = logit.max(3)[0]

        if use_dropout:
            logit = self.dropout(logit)

        out = self.fc_W0(logit)
        return out

    
class Encoder_Decoder(nn.Module):
    def __init__(self, config):
        super(Encoder_Decoder, self).__init__()
        self.config = config
        self.encoder1 = DenseNet()
        
        self.l2r = False
        self.r2l = False
        if self.config.DECODER_TYPE == "L2R":
            self.l2r = True
        elif self.config.DECODER_TYPE == "R2L":
            self.r2l = True
        else:
            self.l2r = True
            self.r2l = True


        if self.l2r: 
            self.init_GRU_model = FcLayer(self.config.D, self.config.N)
            self.emb_model = Embedding(self.config)
            self.gru_model = Gru_cond_layer_aam(self.config)
            self.gru_prob_model = Gru_prob(self.config)

        if self.r2l:
            self.init_GRU_model2 = FcLayer(self.config.D, self.config.N)
            self.emb_model2 = Embedding(self.config)
            self.gru_model2 = Gru_cond_layer_aam(self.config)
            self.gru_prob_model2 = Gru_prob(self.config)


    def forward(self, x, x_mask, y, y_mask, y_reverse,y_mask_reverse,one_step=False):
        # recover permute
        y = y.permute(1, 0)
        y_mask = y_mask.permute(1, 0)
        y_reverse = y_reverse.permute(1, 0)
        y_mask_reverse = y_mask_reverse.permute(1, 0)

        out_mask = x_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        x_mask = out_mask[:, 0::2, 0::2]
        ctx_mask = x_mask
        ctx1 = self.encoder1(x)

        if self.l2r:        
            ctx_mean1 = (ctx1 * ctx_mask[:, None, :, :]).sum(3).sum(2) / ctx_mask.sum(2).sum(1)[:, None]
            init_state1 = self.init_GRU_model(ctx_mean1)

            # two GRU layers
            emb1 = self.emb_model(y)
            h2ts1, cts1, alphas1, _alpha_pasts = self.gru_model(emb1, y_mask, ctx1, ctx_mask, one_step, init_state1, alpha_past=None)
            scores1 = self.gru_prob_model(cts1, h2ts1, emb1, use_dropout=self.config.USE_DROPOUT)
            # permute for multi-GPU training
            alphas1 = alphas1.permute(1, 0, 2, 3)
            scores1 = scores1.permute(1, 0, 2)

        if self.r2l:
            ctx_mean2 = (ctx1 * ctx_mask[:, None, :, :]).sum(3).sum(2) / ctx_mask.sum(2).sum(1)[:, None]
            init_state2 = self.init_GRU_model2(ctx_mean2)  
            # # two GRU layers
            emb2 = self.emb_model2(y_reverse)
            h2ts2, cts2, alphas2, _alpha_pasts = self.gru_model2(emb2, y_mask_reverse, ctx1, ctx_mask, one_step, init_state2, alpha_past=None)
            scores2 = self.gru_prob_model2(cts2, h2ts2, emb2, use_dropout=self.config.USE_DROPOUT)
            # permute for multi-GPU training
            alphas2 = alphas2.permute(1, 0, 2, 3)
            scores2 = scores2.permute(1, 0, 2)

        if self.r2l and self.l2r:
            return scores1, alphas1,scores2, alphas2
        if self.r2l and not self.l2r:
            return scores1, alphas1, None, None
        if not self.r2l and self.l2r:
            return None, None, scores2, alphas2 


    # decoding: encoder part
    def f_init(self, x, x_mask=None,idx_decoder=1):
        if x_mask is None:
            shape = x.shape
            x_mask = torch.ones(shape).cuda()

        out_mask = x_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        x_mask = out_mask[:, 0::2, 0::2]
        ctx_mask = x_mask
        
        ctx1= self.encoder1(x) 
        ctx_mean1 = ctx1.mean(dim=3).mean(dim=2)
        if idx_decoder==1:
            init_state1 = self.init_GRU_model(ctx_mean1)
        elif idx_decoder==2:
            init_state1 = self.init_GRU_model2(ctx_mean1)

        return init_state1,ctx1


    # decoding: decoder part
    def f_next(self, y, y_mask, ctx, ctx_mask, init_state, alpha_past, one_step,idx_decoder=1):

        if idx_decoder == 1:

            emb_beam = self.emb_model(y)

            # one step of two gru layers
            next_state, cts, _alpha, next_alpha_past = self.gru_model(emb_beam, y_mask, ctx, ctx_mask, one_step, init_state, alpha_past)
            # reshape to suit GRU step code
            next_state_ = next_state.view(1, next_state.shape[0], next_state.shape[1])
            cts = cts.view(1, cts.shape[0], cts.shape[1])
            emb_beam = emb_beam.view(1, emb_beam.shape[0], emb_beam.shape[1])
            # calculate probabilities
            scores = self.gru_prob_model(cts, next_state_, emb_beam, use_dropout=self.config.USE_DROPOUT)
            scores = scores.view(-1, scores.shape[2])
            next_probs = F.softmax(scores, dim=1)

        elif idx_decoder ==2:
            emb_beam = self.emb_model2(y)

            # one step of two gru layers
            next_state, cts, _alpha, next_alpha_past = self.gru_model2(emb_beam, y_mask, ctx, ctx_mask, one_step, init_state, alpha_past)
            # reshape to suit GRU step code
            next_state_ = next_state.view(1, next_state.shape[0], next_state.shape[1])
            cts = cts.view(1, cts.shape[0], cts.shape[1])
            emb_beam = emb_beam.view(1, emb_beam.shape[0], emb_beam.shape[1])

            # calculate probabilities
            scores = self.gru_prob_model2(cts, next_state_, emb_beam, use_dropout=self.config.USE_DROPOUT)
            scores = scores.view(-1, scores.shape[2])
            next_probs = F.softmax(scores, dim=1)

        return next_probs, next_state, next_alpha_past,_alpha
    

if __name__=="__main__":
    import os
    import sys
    MODULE_PATH = os.path.abspath(".")
    if MODULE_PATH not in sys.path:
        sys.path.append(MODULE_PATH)

    import config
    
    model=Encoder_Decoder(config=config)
    print(model)
    model.cuda()

    from datasets.datasets import MathFormulaDataset
    data = MathFormulaDataset(config=config,
                              feature_file="data/offline-train.pkl",
                              label_file="data/train_caption.txt",
                              dictionary_file="data/dictionary.txt")
    data_iterator,uid_list=data.data_iterator()

    import time
    alllength=0
    SAVE_ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__),
                                        "../checkpoints")
    for X, Y in data_iterator:
        print(len(X))
        x, x_mask, y_in, y_out, y_mask, y_reverse_in, y_reverse_out, y_reverse_mask=data.prepare_data_bidecoder(X,Y)
        x = torch.from_numpy(x).cuda()
        x_mask = torch.from_numpy(x_mask).cuda()

        #L2R
        y = torch.from_numpy(y_in).cuda()
        y_out = torch.from_numpy(y_out).cuda()
        y_mask = torch.from_numpy(y_mask).cuda()
        y = y.permute(1, 0)
        y_mask = y_mask.permute(1, 0)

        y_reverse = torch.from_numpy(y_reverse_in).cuda()
        y_reverse_out = torch.from_numpy(y_reverse_out).cuda()
        y_reverse_mask = torch.from_numpy(y_reverse_mask).cuda()
        y_reverse = y_reverse.permute(1, 0)
        y_reverse_mask = y_reverse_mask.permute(1, 0)
        torch.onnx.export(model,(x, x_mask, y, y_mask, y_reverse,y_reverse_mask),
                          os.path.join(SAVE_ONNX_MODEL_PATH, "encoder_decoder.onnx"),
                          verbose=True,opset_version=11,export_params=True)
        
        break

    
    
    
    