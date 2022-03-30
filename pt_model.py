import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pt_param import *

logsoftmax= nn.LogSoftmax(dim=-1)
softmax= nn.Softmax(dim=-1)

def init_rnn(rnn):
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)

class PositionalEncoding(nn.Module):
    #https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout= 0.1, max_len = 300):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2.0) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class bilinear_pooling(nn.Module):
    def __init__(self, M=768,N=768,D=256,O=256):
        super(bilinear_pooling, self).__init__()
        self.U1 = nn.Linear(M,D,bias=False)
        self.U2 = nn.Linear(N,D,bias=False)
        self.P = nn.Linear(D,O)
        self.V1 = nn.Linear(M,O,bias=False)
        self.V2 = nn.Linear(N,O,bias=False)

    def forward(self, e1,e2):
        c_=self.P(torch.sigmoid(self.U1(e1))*torch.sigmoid(self.U2(e2)))
        c=c_+self.V1(e1)+self.V2(e2)
        return c

class TransformerModel(nn.Module):
    def __init__(self,input_dim=768,output_dim=5,d_model=256, nhead=4, num_encoder_layers=4,num_decoder_layers=4,dim_feedforward=256,dp = 0.1,device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.bilinear_pooling = bilinear_pooling(M=768,N=768,D=64)
        self.pos_encoder = PositionalEncoding(d_model, dp)
        self.transformer = nn.Transformer(d_model=d_model, 
                                          nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dp,
                                          )
        self.decoder_fc = nn.Linear(output_dim,d_model,bias=False)
        self.out = nn.Linear(d_model, output_dim)
        self.output_dim = output_dim
        self.device = device

    def forward(self, src, src2,tgt,forcing_ratio=1,partial_forcing=True):
        B,seq_len,t_dim = tgt.size()
        pad_mask_src = self.create_pad_mask(src,-1).to(self.device)
        src = self.bilinear_pooling(src,src2)

        if forcing_ratio == 1:
            use_teacher_forcing = True
        elif forcing_ratio > 0:
            if partial_forcing:
                use_teacher_forcing = None  # decide later individually in each step
            else:
                use_teacher_forcing = random.random() < forcing_ratio
        else:
            use_teacher_forcing = False
        
        src = src.permute(1,0,2)
        src = self.pos_encoder(src)    
        src_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)

        if use_teacher_forcing == True:
            tgt = torch.cat((torch.zeros(B,1,t_dim).to(self.device),tgt[:,:-1,:]),1).to(self.device)
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(self.device)
            pad_mask_tgt = self.create_pad_mask(tgt,-1).to(self.device)
            mem_mask = self.create_tri_diag_mask(seq_len).to(self.device)
            tgt=self.decoder_fc(tgt)
            tgt = self.pos_encoder(tgt.permute(1,0,2))
            output = self.transformer(src, tgt, tgt_mask=tgt_mask,src_mask=src_mask,src_key_padding_mask=pad_mask_src,tgt_key_padding_mask=pad_mask_tgt,memory_mask=mem_mask)
            output = self.out(output)
        else:
            memory = self.transformer.encoder(src, src_key_padding_mask=pad_mask_src,mask =src_mask)
            decoder_input = torch.zeros(1,B,t_dim).float().to(self.device)
            output = torch.zeros(seq_len,B,self.output_dim).float().to(self.device)
            for di in range(seq_len):
                pad_mask_tgt = self.create_pad_mask(decoder_input.permute(1,0,2),-1).to(self.device)
                decoder_input_2 = self.pos_encoder(self.decoder_fc(decoder_input))
                # decoder_input_2 = self.pos_encoder(decoder_input) # for hard system
                pred = self.transformer.decoder(decoder_input_2, memory,tgt_key_padding_mask=pad_mask_tgt,memory_key_padding_mask=pad_mask_src) #, memory_mask=mem_mask,tgt_mask=tgt_mask, 
                pred = self.out(pred)
                next_item = pred[-1,:,:].unsqueeze(0)
                output[di,:,:]=next_item
                next_item = softmax(next_item)
                if use_teacher_forcing or (use_teacher_forcing is None and random.random() < forcing_ratio):
                    decoder_input = torch.cat((decoder_input, tgt[:,di,:].unsqueeze(0)), dim=0) # teacher forcing
                else:
                    decoder_input = torch.cat((decoder_input, next_item.detach()), dim=0)
                    # # for hard system:
                    # _,next_item_argmax = torch.max(next_item,-1)
                    # next_item_argmax=next_item_argmax.reshape(1,-1,1).float()
                    # decoder_input = torch.cat((decoder_input, next_item_argmax), dim=0)

        return output.permute(1,0,2)


    def generate_square_subsequent_mask(self,sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def create_pad_mask(self,matrix, pad_token):
        matrix=torch.where(torch.isnan(matrix),torch.tensor(pad_token,dtype=torch.float).to(self.device),matrix)
        return (matrix == pad_token)[:,:,0]

    def create_tri_diag_mask(self,sz):
        a = torch.diag(torch.ones(sz))
        b = torch.diag(torch.ones(sz))
        for i in range(sz):
            for j in range(sz):
                if (j>0 and a[i][j-1]==1) or (i>0 and a[i-1][j]==1):
                    b[i][j]=1
        return b

