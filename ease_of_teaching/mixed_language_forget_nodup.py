from __future__ import print_function
from __future__ import division
import torch
import parser
import os
import random
import copy
import sys
import argparse
import wandb

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class SenderImitation(nn.Module): 
    def __init__(self, args):
        super(SenderImitation, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)

        self.attr2hidden = nn.Linear(self.attrSize, self.hiddenSize).to(self.device)
        self.speaklstm = nn.LSTM(self.vocabSize, self.hiddenSize).to(self.device)  # vocabulary represented by one-hot vector
        self.hidden2vocab = nn.Linear(self.hiddenSize, self.vocabSize).to(self.device)
        # state in lstm
        self.hiddenState, self.cellState = torch.Tensor().to(self.device), torch.Tensor().to(self.device)
        self.criterion = nn.NLLLoss(reduction='mean')
        self.sOptimizer = optim.Adam(self.parameters(), lr=self.sLearnRate)

    def init_hidden(self, batch):
        self.hiddenState = torch.zeros((1, batch, self.hiddenSize), device=self.device)
        self.cellState = torch.zeros((1, batch, self.hiddenSize), device=self.device)

    def forward(self, targets, trg_var):
        attrVector = torch.from_numpy(targets).to(self.device) 
        batch = attrVector.size()[0]  #(batch, vocabSize)
        self.init_hidden(batch)
        # update hidden state as target sender sees
        attrEmbeds = self.attr2hidden(attrVector)  # (batch, hidden)
        self.hiddenState = attrEmbeds.unsqueeze(0)  # initialize target instance as hidden state  (num_layers * num_directions, batch, hidden_size)
        startTokenEmbeds = torch.zeros((1, batch, self.vocabSize), device=self.device)
        lstm_out, (self.hiddenState, self.cellState) = self.speaklstm(startTokenEmbeds, (self.hiddenState, self.cellState))
        # lstm_out  (seq_len, batch, num_directions * hidden_size):
        # save m, speak_log_probs, speak_p_log_p, evaluate_probs
        message = torch.zeros((batch, self.messageLen), device=self.device, dtype=torch.long)
        all_log_probs = []
        for i in range(self.messageLen):
            vocabLogits = self.hidden2vocab(lstm_out).squeeze(0)  # (batch, vocabSize）
            vocabLogProbs = F.log_softmax(vocabLogits, dim=1)  # (batch, vocabSize）
            all_log_probs.append(vocabLogProbs.squeeze(1))
            probs, ch_ind = torch.max(vocabLogProbs, 1)
            message[:, i] = ch_ind
            if i != self.messageLen - 1:
                # convert the generated token to one hot vector, copy as the input to the speakLSTM
                tokenEmbeds = torch.zeros((batch, self.vocabSize), device=self.device)
                chEmbeds = tokenEmbeds.scatter_(1, ch_ind.unsqueeze(1), 1)
                lstm_out, (self.hiddenState, self.cellState) = self.speaklstm(chEmbeds.unsqueeze(0),
                                                                              (self.hiddenState, self.cellState))
                
        all_log_probs = torch.stack(all_log_probs, dim=0)
        trg_var = trg_var.transpose(0, 1).contiguous()  # make time-major [T, B]
        time, batch_size, voc_size = all_log_probs.size()  # time-major!
        log_probs_2d = all_log_probs.contiguous().view(-1, voc_size)
        loss = self.criterion(log_probs_2d, trg_var.view(-1))
        return loss, message, all_log_probs

    def backward(self, loss):
        self.sOptimizer.zero_grad()
        loss.backward()
        self.sOptimizer.step()
        
parser = argparse.ArgumentParser(description='Referential game settings')

parser.add_argument('--gpu', type=int, default=0, help='which gpu if we use gpu')
parser.add_argument('--fname', type=str, default='test', help='folder name to save results')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--jupyter', action='store_true') 
parser.add_argument('--slambda', type=float, default=0.1, help='speaker regularization hyperparameter')
parser.add_argument('--rlambda', type=float, default=0.1, help='listener regularization hyperparameter')
parser.add_argument('--receiverNum', type=int, default=1, help='number of listeners in the population')
parser.add_argument('--topk', type=int, default=3, help='number of top messages when we probe language')
parser.add_argument('--evaluateSize', type=int, default=1000, help='the batch size of test objects when not enumeration')

parser.add_argument('--trainIters', type=int, default=3000, help='number of training iterations')
parser.add_argument('--resetNum', type=int, default=50, help='number of reset iterations')
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--sLearnRate', type=float, default=0.001)
parser.add_argument('--rLearnRate', type=float, default=0.001)
parser.add_argument('--reset_receiver', action='store_true') 
parser.add_argument('--forget_sender', action='store_true') 
parser.add_argument('--same_mask', action='store_true') 
parser.add_argument('--weight_mask', action='store_true') 
parser.add_argument('--reset_to_zero', action='store_true') 
parser.add_argument('--keep_perc', type=float, default=1)
parser.add_argument("--group_vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")
parser.add_argument(
    '--no_wandb',
    action='store_true',
    help='do not write to wandb')
parser.add_argument('--iter_per_batch', type=int, default=1)

args_raw = parser.parse_args()
if not args_raw.no_wandb:
    if len(args_raw.group_vars) > 0:
        if len(args_raw.group_vars) == 1:
            group_name = args_raw.group_vars[0] + str(getattr(args_raw, args_raw.group_vars[0]))
        else:
            group_name = args_raw.group_vars[0] + str(getattr(args_raw, args_raw.group_vars[0]))
            for var in args_raw.group_vars[1:]:
                group_name = group_name + '_' + var + str(getattr(args_raw, var))
        wandb.init(project="mixed_language_forget_v2",
               group=args_raw.fname,
               name=group_name)
        for var in args_raw.group_vars:
            wandb.config.update({var:getattr(args_raw, var)})
            
args = vars(args_raw) # convert python object to dict
# args = parser.parse()  # parsed argument from CLI
args['device'] = torch.device("cuda:" + str(args['gpu']) if torch.cuda.is_available() else "cpu")
if not os.path.exists(args['fname']):
    os.makedirs(args['fname'])

# dataset hyperparameters
args['numColors'] = 8 
args['numShapes'] = 4 
args['attrSize'] = args['numColors'] + args['numShapes']  # colors + shapes

# game settings
args['vocabSize'] = 8
args['messageLen'] = 2
args['distractNum'] = 5  # including targets

# training hyperparameters
args['resetIter'] = args['trainIters'] // args['resetNum']  # life of a receiver: 6K
args['deterResetNums'] = 30
args['deterResetIter'] = 1000

# population of receivers training
args['population'] = False

# model hyperparameters
args['hiddenSize'] = 100 

print(args)

from dataGenerator import Dataset
import utility
import numpy as np
import models
if args['population']:
    from popgame import popGuessGame
else:
    from game import GuessGame

torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
np.random.seed(args['seed'])
random.seed(args['seed'])

torch.backends.cudnn.deterministic=True

def torch_bernoulli(p, size):
    return torch.rand(size) < p
# get data
data = Dataset(args['numColors'], args['numShapes'], args['attrSize'])
train_np = data.getTrain()
util = utility.Utility(args, data)

perfect_sbot = models.overlapPerfectSender(args)
permute_sbot = models.overlapPermutedSender(args)

all_instances = util.dataset.getEnumerateData()
all_instances = torch.from_numpy(all_instances).to(perfect_sbot.device) 
perfect_Message, *rest = perfect_sbot.speak(all_instances, stochastic=False)
permuted_Message, *rest = permute_sbot.speak(all_instances, stochastic=False)

testsender = SenderImitation(args)
teamCLONE = SenderImitation(args)

if args_raw.forget_sender and args_raw.same_mask:
    if args_raw.weight_mask:
        mask_dict = {}
        for name, param in testsender.named_parameters():
            weight_mag = torch.abs(param.detach().clone())
            topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement()*(1-args_raw.keep_perc)), largest=False)
            temp_mask = torch.ones(weight_mag.nelement())
            temp_mask[topk.indices] = 0
            mask_dict[name] = temp_mask.bool().view(weight_mag.shape)
    else:
        mask_dict = {}
        for name, param in testsender.named_parameters():
            mask_dict[name] = torch_bernoulli(args_raw.keep_perc, param.shape)

for i in range(args['trainIters']):
#     candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'])
#     target_ids = np.argmax(targets @ all_instances.cpu().numpy().T, 1)
#     batch_size = targets.shape[0]
    target_ids = np.random.choice(32,32,False)
    targets = all_instances[target_ids].cpu().numpy()
    batch_size = 32
    true_messages = torch.cat([perfect_Message[target_ids[:batch_size//2]],permuted_Message[target_ids[batch_size//2:]]])
#     true_messages = target_Message[target_ids]
    for k in range(args_raw.iter_per_batch):
        loss, message, all_log_probs = testsender(targets, true_messages)
        testsender.backward(loss)
    if i % 100 == 0:
        with torch.no_grad():
            true_1 = torch.mean(torch.sum(-torch.exp(all_log_probs[0, :batch_size//2, :]) * all_log_probs[0, :batch_size//2, :],1))
            true_2 = torch.mean(torch.sum(-torch.exp(all_log_probs[1, :batch_size//2, :]) * all_log_probs[1, :batch_size//2, :],1))
            perm_1 = torch.mean(torch.sum(-torch.exp(all_log_probs[0, batch_size//2:, :]) * all_log_probs[0, batch_size//2:, :],1))
            perm_2 = torch.mean(torch.sum(-torch.exp(all_log_probs[1, batch_size//2:, :]) * all_log_probs[1, batch_size//2:, :],1))
        record = 'Iteration ' + str(i) \
                 + ' Sender loss ' + str(np.round(loss.item(), decimals=4)) \
                 + ' Exact Match Perfect ' + str(np.round(torch.mean((torch.sum(true_messages == message,1)[:batch_size//2] == 2).float()).item()*100, decimals=2)) \
                 + ' Exact Match Permuted ' + str(np.round(torch.mean((torch.sum(true_messages == message,1)[batch_size//2:] == 2).float()).item()*100, decimals=2)) \
                 + ' Exact Match ' + str(np.round(torch.mean((torch.sum(true_messages == message,1) == 2).float()).item()*100, decimals=2)) + '%\n'
        print(record)        
        if not args_raw.no_wandb:
            wandb.log({'Iter': i, 'Sender loss': np.round(loss.item(), decimals=4), 'Exact Match Perfect':np.round(torch.mean((torch.sum(true_messages == message,1)[:batch_size//2] == 2).float()).item()*100, decimals=2), 'Exact Match Permuted':np.round(torch.mean((torch.sum(true_messages == message,1)[batch_size//2:] == 2).float()).item()*100, decimals=2), 'Exact Match Overall':np.round(torch.mean((torch.sum(true_messages == message,1) == 2).float()).item()*100, decimals=2), 'True 1 Entropy':np.round(true_1.cpu().numpy(), decimals=4), 'True 2 Entropy':np.round(true_2.cpu().numpy(), decimals=4), 'Perm 1 Entropy':np.round(perm_1.cpu().numpy(), decimals=4), 'Perm 2 Entropy':np.round(perm_2.cpu().numpy(), decimals=4)})
        
        teamCLONE.load_state_dict(testsender.state_dict())
        teamCLONE.load_state_dict(testsender.state_dict())
        
        if not args_raw.same_mask:
            if args_raw.weight_mask:
                mask_dict = {}
                for name, param in teamCLONE.named_parameters():
                    weight_mag = torch.abs(param.detach().clone())
                    topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement()*(1-args_raw.keep_perc)), largest=False)
                    temp_mask = torch.ones(weight_mag.nelement())
                    temp_mask[topk.indices] = 0
                    mask_dict[name] = temp_mask.bool().view(weight_mag.shape)
            else:
                mask_dict = {}
                for name, param in teamCLONE.named_parameters():
                        mask_dict[name] = torch_bernoulli(args_raw.keep_perc, param.shape)
        if args_raw.reset_to_zero:
            for name, param in teamCLONE.named_parameters():
                param.data *= mask_dict[name].float().to('cuda')
        else:
            metric_dict = {}
            for name, param in teamCLONE.named_parameters():
                metric_dict[name] = param.detach().clone()

            for name, param in teamCLONE.named_children():
                if name != 'criterion':
                    param.reset_parameters()

            for name, param in teamCLONE.named_parameters():
                param.data[mask_dict[name]] = metric_dict[name][mask_dict[name]]
                
        with torch.no_grad():
            loss, message, all_log_probs = teamCLONE(targets, true_messages)
            true_1 = torch.mean(torch.sum(-torch.exp(all_log_probs[0, :batch_size//2, :]) * all_log_probs[0, :batch_size//2, :],1))
            true_2 = torch.mean(torch.sum(-torch.exp(all_log_probs[1, :batch_size//2, :]) * all_log_probs[1, :batch_size//2, :],1))
            perm_1 = torch.mean(torch.sum(-torch.exp(all_log_probs[0, batch_size//2:, :]) * all_log_probs[0, batch_size//2:, :],1))
            perm_2 = torch.mean(torch.sum(-torch.exp(all_log_probs[1, batch_size//2:, :]) * all_log_probs[1, batch_size//2:, :],1))
            record = 'Iteration ' + str(i) \
                     + ' COPY Sender loss ' + str(np.round(loss.item(), decimals=4)) \
                     + ' COPY Exact Match Perfect ' + str(np.round(torch.mean((torch.sum(true_messages == message,1)[:batch_size//2] == 2).float()).item()*100, decimals=2)) \
                     + ' COPY Exact Match Permuted ' + str(np.round(torch.mean((torch.sum(true_messages == message,1)[batch_size//2:] == 2).float()).item()*100, decimals=2)) \
                     + ' COPY Exact Match ' + str(np.round(torch.mean((torch.sum(true_messages == message,1) == 2).float()).item()*100, decimals=2)) + '%\n'
            print(record)
            if not args_raw.no_wandb:
                wandb.log({'Iter': i, 'Copy Sender loss': np.round(loss.item(), decimals=4), 'Copy Exact Match Perfect':np.round(torch.mean((torch.sum(true_messages == message,1)[:batch_size//2] == 2).float()).item()*100, decimals=2), 'Copy Exact Match Permuted':np.round(torch.mean((torch.sum(true_messages == message,1)[batch_size//2:] == 2).float()).item()*100, decimals=2), 'Copy Exact Match Overall':np.round(torch.mean((torch.sum(true_messages == message,1) == 2).float()).item()*100, decimals=2), 'Copy True 1 Entropy':np.round(true_1.cpu().numpy(), decimals=4), 'Copy True 2 Entropy':np.round(true_2.cpu().numpy(), decimals=4), 'Copy Perm 1 Entropy':np.round(perm_1.cpu().numpy(), decimals=4), 'Copy Perm 2 Entropy':np.round(perm_2.cpu().numpy(), decimals=4)})
                
if not args_raw.no_wandb:
    wandb.run.finish()
