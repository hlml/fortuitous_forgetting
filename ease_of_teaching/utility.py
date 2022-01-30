# @title Utility functions
from __future__ import print_function
from __future__ import division
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import numpy as np
import scipy
from scipy import spatial
from scipy import stats
import itertools
#from builtins import dict


class Utility():
    def __init__(self, args, data):
        for key, value in args.items():
            setattr(self, key, value)
        # no more than 8 colors and 5 shapes
        self.dataset = data
        self.ch_vocab = {i: chr(i + 97) for i in range(self.vocabSize)}
        self.targets_to_attr_dict = self.get_targets_to_attr_dict()
        self.message_dict = self.get_idx_to_message_dict()
        self.targetDist = self.get_cos_between_targets_dict()
        self.msgDist = self.get_levenshtein_dict()

    def get_targets_to_attr_dict(self):
        targets_to_attr_d = {}
        colorD = {0: 'black', 1: 'blue', 2: 'green', 3: 'grey', 4: 'pink', 5: 'purple', 6: 'red', 7: 'yellow'}
        shapeD = {0: 'circle', 1: 'square', 2: 'star', 3: 'triangle'}
        all_combinations_targets = self.dataset.getEnumerateData()  # args['numColors'] * args['numShapes']
        for attrVector in all_combinations_targets:
            for i in range(self.numColors):
                if attrVector[i] == 1:
                    attrColor = colorD[i]
                    break
            for j in range(self.numColors, self.numColors + self.numShapes):
                if attrVector[j] == 1:
                    attrShape = shapeD[j - self.numColors]
                    break
            targets_to_attr_d[tuple(attrVector)] = (attrColor, attrShape)
        print('Generated targets to attribute dictionary of size: ', len(targets_to_attr_d))
        return targets_to_attr_d

    def get_idx_to_message_dict(self):
        m_vocab = {}
        for tuple in itertools.product(self.ch_vocab, repeat=self.messageLen):
            mes = ''
            for ch in tuple:
                mes += self.ch_vocab[ch]
            m_vocab[tuple] = mes
        print('Generated index to message dictionary of size: ', len(m_vocab))
        return m_vocab

    def probeLanguage(self, message, targets, speakProbs=None, predProbs=None, rewards=None):
        # print('targets', np.shape(targets)) #(batch, AttrSize)
        # print('message', np.shape(message)) #(batch, messageLen)
        # probe language on train dataset:
        attrD = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # d[color][shape][message]=frequency
        if speakProbs is not None:
            attrP = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # d[color][shape][message]=totalProbs
        if predProbs is not None:
            attrPp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # d[color][shape][message]=totalProbs
        if rewards is not None:
            correctD = defaultdict(
                lambda: defaultdict(lambda: defaultdict(int)))  # d[color][shape][message]=numOfCorrect
        m_str = []
        for m in message:
            m_app = self.message_dict[tuple(m.numpy())]
            m_str.append(m_app)
        for ind, t in enumerate(targets):
            attrVector = targets[ind][:self.numColors + self.numShapes]
            attrColor, attrShape = self.targets_to_attr_dict[tuple(attrVector)]
            attrD[attrColor][attrShape][m_str[ind]] += 1
            if speakProbs is not None:
                attrP[attrColor][attrShape][m_str[ind]] += speakProbs[ind]
            if predProbs is not None:
                attrPp[attrColor][attrShape][m_str[ind]] += predProbs[ind]
            if rewards is not None and rewards.data[ind] == 1.0:
                correctD[attrColor][attrShape][m_str[ind]] += 1

        attrtableD = defaultdict(lambda: defaultdict(list))  # d[shape][color] = [(message, frequency)] correct
        for c in attrD:
            for s in attrD[c]:
                total = 0
                attrSpeakProbs = defaultdict(float)
                for m in attrD[c][s]:
                    total += attrD[c][s][m]
                    if speakProbs is not None:
                        attrSpeakProbs[m] = attrP[c][s][m] / attrD[c][s][m]  # probablity / frequency
                    else:
                        attrSpeakProbs[m] = attrD[c][s][m]
                sortL = sorted(attrSpeakProbs, key=attrSpeakProbs.get, reverse=True)  # sort by speak probability
                topind = min(len(sortL), self.topk)

                for m in sortL[:topind]:
                    if predProbs is not None and rewards is not None:
                        attrtableD[s][c].append((m, attrD[c][s][m], '%.4f' % (attrP[c][s][m] / attrD[c][s][m]),
                                                 '%.4f' % (attrPp[c][s][m] / attrD[c][s][m]),
                                                 '%d' % (correctD[c][s][m] / attrD[c][s][m] * 100) + '%',
                                                 '%d' % (attrD[c][s][m] / total * 100) + '%'))
                    elif speakProbs is not None:
                        attrtableD[s][c].append((m, '%.4f' % (attrP[c][s][m] / attrD[c][s][m])))  # (message, speak_probs)
                    else:
                        attrtableD[s][c].append(m) #(message only)
        return attrtableD

    def drawTable(self, trainD):
        zD = deepcopy(trainD)
        df = pd.DataFrame.from_dict(zD)
        return df

    def levenshtein(self, s, t):
        """
            levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein
            distance between the first i characters of s and the
            first j characters of t
        """
        rows = len(s) + 1
        cols = len(t) + 1
        dist = [[0 for x in range(cols)] for x in range(rows)]
        # source prefixes can be transformed into empty strings
        # by deletions:
        for i in range(1, rows):
            dist[i][0] = i
        # target prefixes can be created from an empty source string
        # by inserting the characters
        for i in range(1, cols):
            dist[0][i] = i

        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[row][col] = min(dist[row - 1][col] + 1,  # deletion
                                     dist[row][col - 1] + 1,  # insertion
                                     dist[row - 1][col - 1] + cost)  # substitution
        return dist[row][col]

    def get_levenshtein_dict(self):
        levenshtein_dict = {}
        for s in self.message_dict.values():
            for t in self.message_dict.values():
                levenshtein_dict[(s,t)] = self.levenshtein(s, t)
        print('Generated dictionary of levenshtein_distance between messages of size: ', len(levenshtein_dict))
        return levenshtein_dict

    def get_cos_between_targets_dict(self):
        cos_target_dict = {}
        all_combinations_targets = self.dataset.getEnumerateData()  # args['numColors'] * args['numShapes']
        for i in all_combinations_targets:
            for j in all_combinations_targets:
                cos_target_dict[(tuple(i), tuple(j))] = -scipy.spatial.distance.cosine(i, j) + 1
        print('Generated dictionary of cosine similarity between targets of size: ', len(cos_target_dict))
        return cos_target_dict

    def topographicMeasure(self, message, targets):
        # generate message
        messageL = []
        for m in message:
            m_app = self.message_dict[tuple(m.numpy())]
            messageL.append(m_app)
        # calculate Levenshtein distances between all pairs of objects' messages
        Ldistance = np.zeros(len(messageL) * len(messageL))
        for x, i in enumerate(messageL):
            for y, j in enumerate(messageL):
                Ldistance[x * len(messageL) + y] = self.msgDist[(i, j)]  # only for category data
        # calculate cosine similarity between all pairs of vectors
        import scipy
        cosSimilarity = np.zeros(len(targets) * len(targets))
        for x, i in enumerate(targets):
            for y, j in enumerate(targets):
                cosSimilarity[x * len(targets) + y] = self.targetDist[(tuple(i), tuple(j))]
        topographicM, pValue = scipy.stats.spearmanr(Ldistance, cosSimilarity)
        return -topographicM, pValue

    def get_sender_language(self, team, neural):
        all_instances = self.dataset.getEnumerateData()
        eMessage, deter_entropy, speak_probs = team.senderForward(all_instances, neural)

        topoM, pValue = self.topographicMeasure(eMessage.cpu(), all_instances)
#         print('Topographic Messure for all instance combinations is ', '{:.4f}'.format(topoM), 'and p value is ', pValue, '\n')

#         print('The language the sender is speaking: ')
        trainD = self.probeLanguage(eMessage.cpu(), all_instances, speak_probs.cpu())
#         df = self.drawTable(trainD)
#         if self.jupyter:
#             display(df)
#         else:
#             print(df.to_string())
#         print()
        return topoM, deter_entropy, dict(trainD)
