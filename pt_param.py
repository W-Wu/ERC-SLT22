from argparse import Namespace

param =  Namespace()
param.seed=929
param.train_batch_size=100
param.learning_rate = 5e-4
param.EPOCH = 1#5
param.l2_reg = 0.001
param.scheduler = "adam"
param.output_dim=5
param.grad_accum = 1

param.partial_forcing = True 
param.forcing_type = 'exp'
param.forcing_ratio_default = 0.75
param.forcing_decay_default =0.9
param.epoch_start_tf =1

param.kl=20

param.save_model = True