# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math

class Gru_cond_layer_aam(nn.Module):
    def __init__(self, config):
        super(Gru_cond_layer_aam, self).__init__()
        self.config = config
        # attention
        self.conv_Ua = nn.Conv2d(self.config.D, self.config.DIM_ATTENTION, kernel_size=1)
        self.fc_Wa = nn.Linear(self.config.N, self.config.DIM_ATTENTION, bias=False)
        self.conv_Q = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)

        #5x5
        self.conv_Q2 = nn.Conv2d(1, 512, kernel_size=5, bias=False, padding=2)


        self.fc_Uf = nn.Linear(512, self.config.DIM_ATTENTION)
        self.fc_Uf2 = nn.Linear(512, self.config.DIM_ATTENTION)
        self.fc_va = nn.Linear(self.config.DIM_ATTENTION, 1)

        # the first GRU layer
        self.fc_Wyz = nn.Linear(self.config.M, self.config.N)
        self.fc_Wyr = nn.Linear(self.config.M, self.config.N)
        self.fc_Wyh = nn.Linear(self.config.M, self.config.N)

        self.fc_Uhz = nn.Linear(self.config.N, self.config.N, bias=False)
        self.fc_Uhr = nn.Linear(self.config.N, self.config.N, bias=False)
        self.fc_Uhh = nn.Linear(self.config.N, self.config.N, bias=False)

        # the second GRU layer
        self.fc_Wcz = nn.Linear(self.config.D, self.config.N, bias=False)
        self.fc_Wcr = nn.Linear(self.config.D, self.config.N, bias=False)
        self.fc_Wch = nn.Linear(self.config.D, self.config.N, bias=False)

        self.fc_Uhz2 = nn.Linear(self.config.N, self.config.N)
        self.fc_Uhr2 = nn.Linear(self.config.N, self.config.N)
        self.fc_Uhh2 = nn.Linear(self.config.N, self.config.N)
        
    def forward(self, embedding, mask=None, context=None, context_mask=None, one_step=False, init_state=None,
                alpha_past=None):
        n_steps = embedding.shape[0]
        n_samples = embedding.shape[1]

        Ua_ctx = self.conv_Ua(context)
        Ua_ctx = Ua_ctx.permute(2, 3, 0, 1) 
        state_below_z = self.fc_Wyz(embedding)
        state_below_r = self.fc_Wyr(embedding)
        state_below_h = self.fc_Wyh(embedding)

        if one_step:
            if mask is None:
                mask = torch.ones(embedding.shape[0]).cuda()
            h2ts, cts, alphas, alpha_pasts = self._step_slice(mask, state_below_r, state_below_z, state_below_h,
                                                              init_state, context, context_mask, alpha_past, Ua_ctx)
        else:
            alpha_past = torch.zeros(n_samples, context.shape[2], context.shape[3]).cuda()
            h2t = init_state
            h2ts = torch.zeros(n_steps, n_samples, self.config.N).cuda()
            cts = torch.zeros(n_steps, n_samples, self.config.D).cuda()
            alphas = (torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3])).cuda()
            alpha_pasts = torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3]).cuda()
            for i in range(n_steps):
                h2t, ct, alpha, alpha_past = self._step_slice(mask[i], state_below_r[i], state_below_z[i],
                                                              state_below_h[i], h2t, context, context_mask, alpha_past,
                                                              Ua_ctx)
                h2ts[i] = h2t
                cts[i] = ct
                alphas[i] = alpha
                alpha_pasts[i] = alpha_past
        return h2ts, cts, alphas, alpha_pasts
    
    def _step_slice(self, mask, state_below_r, state_below_z, state_below_h, h, ctx, ctx_mask, alpha_past, Ua_ctx):
        """
        one step of two GRU layers
        """
        # the first GRU layer
        z1 = torch.sigmoid(self.fc_Uhz(h) + state_below_z)
        r1 = torch.sigmoid(self.fc_Uhr(h) + state_below_r)
        h1_p = torch.tanh(self.fc_Uhh(h) * r1 + state_below_h)
        h1 = z1 * h + (1. - z1) * h1_p
        h1 = mask[:, None] * h1 + (1. - mask)[:, None] * h

        # attention
        Wa_h1 = self.fc_Wa(h1)
        alpha_past_ = alpha_past[:, None, :, :]
        cover_F = self.conv_Q(alpha_past_).permute(2, 3, 0, 1)
        cover_vector = self.fc_Uf(cover_F)

        cover_F2 = self.conv_Q2(alpha_past_).permute(2, 3, 0, 1)
        cover_vector2 = self.fc_Uf2(cover_F2)

        attention_score = torch.tanh(Ua_ctx + Wa_h1[None, None, :, :] + cover_vector+cover_vector2)


        alpha = self.fc_va(attention_score)
        alpha = alpha.view(alpha.shape[0], alpha.shape[1], alpha.shape[2])
        alpha = torch.exp(alpha)
        if (ctx_mask is not None):
            alpha = alpha * ctx_mask.permute(1, 2, 0)
        alpha = alpha / alpha.sum(1).sum(0)[None, None, :]
        alpha_past = alpha_past + alpha.permute(2, 0, 1)
        ct = (ctx * alpha.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)

        # the second GRU layer
        z2 = torch.sigmoid(self.fc_Wcz(ct) + self.fc_Uhz2(h1))
        r2 = torch.sigmoid(self.fc_Wcr(ct) + self.fc_Uhr2(h1))
        h2_p = torch.tanh(self.fc_Wch(ct) + self.fc_Uhh2(h1) * r2)
        h2 = z2 * h1 + (1. - z2) * h2_p
        h2 = mask[:, None] * h2 + (1. - mask)[:, None] * h1


        return h2, ct, alpha.permute(2, 0, 1), alpha_past