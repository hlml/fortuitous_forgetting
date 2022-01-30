from __future__ import print_function
from __future__ import division
import torch
import torch.optim as optim
import models
import numpy as np
import copy

class GuessGame:
    def __init__(self, args):
        self.args = args
        for key, value in args.items():
            setattr(self, key, value)
        self.sbot = models.Sender(args).to(self.device)
        print('sbot', self.sbot)

        self.rbot = models.Receiver(args).to(self.device)
        print('rbot', self.rbot)

        self.sOptimizer = optim.Adam(self.sbot.parameters(), lr=self.sLearnRate, weight_decay=self.weight_decay)
        self.rOptimizer = optim.Adam(self.rbot.parameters(), lr=self.rLearnRate, weight_decay=self.weight_decay)

    def forward(self, targets, candidates, evaluate=False, sOpt=True, rOpt=True, stochastic=True): # sOpt equals false only for non-nn models
        targetsTensor = torch.from_numpy(targets).to(self.device) 
        batch = targetsTensor.size()[0]

        if sOpt:  # if sbot == False, it may not be a neural network agent
            self.sbot.init_hidden(batch)
        if rOpt:
            self.rbot.init_hidden(batch)

        if evaluate:
            self.sbot.eval()  
            self.rbot.eval()
        else:
            self.sbot.train()
            self.rbot.train()

        m, speak_log_probs, speak_p_log_p, evaluate_probs = self.sbot.speak(targetsTensor, stochastic)

        self.rbot.listen(m)

        candidatesTensor = torch.from_numpy(candidates).to(self.device)
        p_action, pred_log_probs, pred_p_log_p, pred_probs = self.rbot.predict(candidatesTensor, stochastic)

        predicts = candidatesTensor[np.arange(batch), p_action, :]
        mattersIndex = self.numColors + self.numShapes
        tsum = torch.LongTensor(batch).fill_(mattersIndex).to(self.device)
        psum = torch.eq(predicts[:, 0:mattersIndex], targetsTensor[:, 0:mattersIndex]).sum(1)  # true if predicts and targets match at every index

        rewards = torch.eq(tsum, psum).float()

        sloss = -rewards * speak_log_probs + self.slambda * speak_p_log_p
        if stochastic:
            rloss = -rewards * pred_log_probs + self.rlambda * pred_p_log_p
        else:
            rloss = -rewards * pred_log_probs + 0.1 * pred_p_log_p
        sloss = torch.sum(sloss) / batch
        rloss = torch.sum(rloss) / batch

        batch_entropy = -torch.sum(speak_p_log_p) / batch
        return sloss, rloss, m, rewards, batch_entropy, evaluate_probs, pred_probs # evaluate_probs, pred_probs for probing languages

    def backward(self, sloss, rloss, sOpt=True, rOpt=True): # sOpt is false when we do not need update sender
        if sOpt:
            self.sOptimizer.zero_grad()
            sloss.backward()
            self.sOptimizer.step()
        if rOpt:
            self.rOptimizer.zero_grad()
            rloss.backward()
            self.rOptimizer.step()

    def senderForward(self, targets, neural):
        targetsTensor = torch.from_numpy(targets).to(self.device) 
        batch = targetsTensor.size()[0]

        if neural:
            self.sbot.init_hidden(batch)
            self.sbot.eval()

        m, _, p_log_p, speak_probs = self.sbot.speak(targetsTensor, stochastic=False)

        deter_entropy = -torch.sum(p_log_p) / batch

        return m, deter_entropy, speak_probs

    def freezeSender(self):
        for param in self.sbot.parameters():
            param.requires_grad = False
        print('\nSender parameters are freezed now')

    def defreezeSender(self):
        for param in self.sbot.parameters():
            param.requires_grad = True
        print('\nSender parameters are trainable now')

    def freezeReceiver(self):
        for param in self.rbot.parameters():
            param.requires_grad = False
        print('\nReceiver parameters are freezed now')

    def defreezeReceiver(self):
        for param in self.rbot.parameters():
            param.requires_grad = True
        print('\nReceiver parameters are trainable now')

        
    def copyReceiver(self):
        oldbot = models.Receiver(self.args).to(self.device)
        oldbot.load_state_dict(self.rbot.state_dict())
        print('\nParameters in the old Receiver are copied')
        oldrOptimizer = optim.Adam(oldbot.parameters(), lr=self.rLearnRate, weight_decay=self.weight_decay)
        oldrOptimizer.load_state_dict(self.rOptimizer.state_dict())
        return oldbot, oldrOptimizer

    def resetReceiver(self, sOpt=True): # sOpt could be false, when sender only used for evaluation
        self.rbot.attr2embed.reset_parameters()
        self.rbot.listenlstm.reset_parameters()
        self.rbot.hidden2embed.reset_parameters()
        print('\nParameters in the Receiver are reset')

        self.rOptimizer = optim.Adam(self.rbot.parameters(), lr=self.rLearnRate, weight_decay=self.weight_decay)
        print('Reinitialize receiver optimizer')
        if sOpt:
            self.sOptimizer = optim.Adam(self.sbot.parameters(), lr=self.sLearnRate)
            print('Reinitialize sender optimizer')
            
    def resetSender(self, rOpt=True): # sOpt could be false, when sender only used for evaluation
        self.sbot.attr2hidden.reset_parameters()
        self.sbot.speaklstm.reset_parameters()
        self.sbot.hidden2vocab.reset_parameters()
        print('\nParameters in the Sender are reset')

        self.sOptimizer = optim.Adam(self.sbot.parameters(), lr=self.sLearnRate, weight_decay=self.weight_decay)
        print('Reinitialize sender optimizer')
        if rOpt:
            self.rOptimizer = optim.Adam(self.rbot.parameters(), lr=self.rLearnRate, weight_decay=self.weight_decay)
            print('Reinitialize receiver optimizer')

