#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @title Model
from __future__ import print_function
from __future__ import division
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class Sender(nn.Module): 
    def __init__(self, args):
        super(Sender, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)

        self.attr2hidden = nn.Linear(self.attrSize, self.hiddenSize).to(self.device)
        self.speaklstm = nn.LSTM(self.vocabSize, self.hiddenSize).to(self.device)  # vocabulary represented by one-hot vector
        self.hidden2vocab = nn.Linear(self.hiddenSize, self.vocabSize).to(self.device)
        # state in lstm
        self.hiddenState, self.cellState = torch.Tensor().to(self.device), torch.Tensor().to(self.device)

    def init_hidden(self, batch):
        self.hiddenState = torch.zeros((1, batch, self.hiddenSize), device=self.device)
        self.cellState = torch.zeros((1, batch, self.hiddenSize), device=self.device)

    def speak(self, attrVector, stochastic=True):
        batch = attrVector.size()[0]  #(batch, vocabSize)

        # update hidden state as target sender sees
        attrEmbeds = self.attr2hidden(attrVector)  # (batch, hidden)
        self.hiddenState = attrEmbeds.unsqueeze(0)  # initialize target instance as hidden state  (num_layers * num_directions, batch, hidden_size)
        startTokenEmbeds = torch.zeros((1, batch, self.vocabSize), device=self.device)
        lstm_out, (self.hiddenState, self.cellState) = self.speaklstm(startTokenEmbeds, (self.hiddenState, self.cellState))
        # lstm_out  (seq_len, batch, num_directions * hidden_size):
        # save m, speak_log_probs, speak_p_log_p, evaluate_probs
        message = torch.zeros((batch, self.messageLen), device=self.device, dtype=torch.long)
        speak_log_probs = torch.zeros(batch, device=self.device)
        p_log_p = torch.zeros(batch, device=self.device)
        evaluate_probs = torch.ones(batch, device=self.device)

        for i in range(self.messageLen):
            vocabLogits = self.hidden2vocab(lstm_out).squeeze(0)  # (batch, vocabSize）
            vocabProbs = F.softmax(vocabLogits, dim=1)  # (batch, vocabSize）
            vocabLogProbs = F.log_softmax(vocabLogits, dim=1)  # (batch, vocabSize）
            p_log_p.add_(torch.sum(vocabProbs * vocabLogProbs, -1))  # p_log_p: negative entropy
            if stochastic:
                vocabDistr = D.Categorical(vocabProbs)
                ch_ind = vocabDistr.sample()
                logP = vocabDistr.log_prob(ch_ind)
                speak_log_probs.add_(logP)
                evaluate_probs.mul_(torch.exp(logP))
            else:
                probs, ch_ind = torch.max(vocabProbs, 1)
                evaluate_probs.mul_(probs)
                speak_log_probs.add_(torch.log(probs))
            message[:, i] = ch_ind
            if i != self.messageLen - 1:
                # convert the generated token to one hot vector, copy as the input to the speakLSTM
                tokenEmbeds = torch.zeros((batch, self.vocabSize), device=self.device)
                chEmbeds = tokenEmbeds.scatter_(1, ch_ind.unsqueeze(1), 1)
                lstm_out, (self.hiddenState, self.cellState) = self.speaklstm(chEmbeds.unsqueeze(0),
                                                                              (self.hiddenState, self.cellState))
        return message, speak_log_probs, p_log_p, evaluate_probs  # return character index in vocab

class Receiver(nn.Module):
    def __init__(self, args):
        super(Receiver, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)

        self.attr2embed = nn.Linear(self.attrSize, self.hiddenSize).to(self.device)
        self.listenlstm = nn.LSTM(self.vocabSize, self.hiddenSize).to(self.device)
        self.hidden2embed = nn.Linear(self.hiddenSize, self.hiddenSize).to(self.device)

        self.hiddenState, self.cellState = torch.Tensor().to(self.device), torch.Tensor().to(self.device)

    def init_hidden(self, batch):
        self.hiddenState = torch.zeros((1, batch, self.hiddenSize), device=self.device)
        self.cellState = torch.zeros((1, batch, self.hiddenSize), device=self.device)

    def listen(self, message):  # message (batch, message_len)
        batch = message.size()[0]
        # convert message token to one hot vector
        chEmbeds = torch.zeros((batch * self.messageLen, self.vocabSize), device=self.device)
        s_message = message.view(-1, 1)
        tokenembeds = chEmbeds.scatter_(1, s_message, 1)
        tokenembeds = tokenembeds.view(batch, self.messageLen, self.vocabSize)  # (batch, self.messageLen, self.vocabSize)
        lstm_out, (self.hiddenState, self.cellState) = self.listenlstm(torch.transpose(tokenembeds, 0, 1), (self.hiddenState, self.cellState))

    def predict(self, distrImages, stochastic=True):
        outEmbeds = self.hidden2embed(torch.squeeze(self.hiddenState, 0))  # (batch, embedding_dim)
        distraEmbeds = self.attr2embed(distrImages)  # (batch, Kimages, embedding_dim)
        out = torch.matmul(torch.unsqueeze(torch.unsqueeze(outEmbeds, 1), 2), torch.unsqueeze(distraEmbeds, 3))
        # (batch, Kimages, 1, 1)
        out = torch.squeeze(out)

        outProbs = F.softmax(out, 1) #(batch, Kimages)
        outLogits = F.log_softmax(out, 1)
        p_log_p = torch.sum(outLogits * outProbs, -1)  # (batch)
        if stochastic:
            outDistr = D.Categorical(outProbs)
            action_ind = outDistr.sample()  # (batch)
            saved_log_probs = outDistr.log_prob(action_ind)
            pred_probs = torch.exp(saved_log_probs)
        else:
            pred_probs, action_ind = torch.max(outProbs, 1)  # (batch)
            saved_log_probs = torch.log(pred_probs)
        return action_ind, saved_log_probs, p_log_p, pred_probs

class overlapPerfectSender(nn.Module):
    def __init__(self, args):
        super(overlapPerfectSender, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)

            #     colorD = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'pink', 5: 'grey', 6: 'purple', 7: 'black'}
            #     shapeD = {0: 'circle', 1: 'square', 2: 'triangle', 3: 'star'}
            #     vocab = {i: chr(i + 97) for i in range(self.vocabSize)}
        print('Sbot is a overlapPerfectSender')

    def speak(self, attrVector, stochastic=False):
        batch = attrVector.size()[0]

        attrMessages = torch.nonzero(attrVector[:, :self.numColors + self.numShapes])
        messages = attrMessages[:, 1].contiguous().view(batch, self.messageLen)

        mask = torch.ge(messages, self.numColors)
        messages.masked_scatter_(mask, messages.masked_select(mask) - self.numColors)

        return messages, torch.zeros(batch, device=self.device), torch.zeros(batch, device=self.device), torch.ones(batch, device=self.device)


class overlapPermutedSender(nn.Module):
    def __init__(self, args, perm=None):
        super(overlapPermutedSender, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)

        #     colorD = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'pink', 5: 'grey', 6: 'purple', 7: 'black'}
        #     shapeD = {0: 'circle', 1: 'square', 2: 'triangle', 3: 'star'}
        #     vocab = {i: chr(i + 97) for i in range(self.vocabSize)}
        print('Sbot is a overlapPermutedSender')

        vocab = [[i, j] for i in range(self.numColors) for j in range(self.numShapes)]
        vocabTensor = torch.tensor(vocab, dtype=torch.long, device=self.device)

        if perm is None:
            perm = torch.randperm(self.numColors * self.numShapes, device=self.device)
        permVocab = torch.index_select(vocabTensor, 0, perm)

        keys = []
        for m in vocab:
            keys.append(''.join(map(str, m)))  # key is in format '60'

        self.vocab_d = dict(zip(keys, permVocab)) # message to attrbutes tensor

    def speak(self, attrVector, stochastic=False):
        batch = attrVector.size()[0]

        attrMessages = torch.nonzero(attrVector[:, :self.numColors + self.numShapes])
        messages = attrMessages[:, 1].contiguous().view(batch, self.messageLen)
        # minus numColors
        mask = torch.ge(messages, self.numColors)
        messages.masked_scatter_(mask, messages.masked_select(mask) - self.numColors)
        # find actual used vocab in the permuted dict
        permMessages = torch.zeros((batch, self.messageLen), dtype=torch.long, device=self.device)
        for ind, m in enumerate(messages):
            permMessages[ind] = self.vocab_d[''.join(map(str, m.detach().tolist()))]

        return permMessages, torch.zeros(batch, device=self.device), torch.zeros(batch, device=self.device), torch.ones(batch, device=self.device)

    def senderForward(self, targets):
        targetsTensor = torch.from_numpy(targets).to(self.device) 
        batch = targetsTensor.size()[0]

        m, _, _, speak_probs = self.speak(targetsTensor, True)

        return m, speak_probs


