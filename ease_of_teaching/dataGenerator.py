# Generate data
import numpy as np
import random


class Dataset:
    def __init__(self, numColors, numShapes, attrSize):
        self.numColors = numColors  # number of colors appeared in the dataset
        self.numShapes = numShapes  # number of shapes appeared in the dataset
        self.attrSize = attrSize  # bits of features
        self.train_np = np.array([])

    def getTrain(self): 
        trainInd = [(i, j) for i in range(self.numColors) for j in range(self.numShapes)]

        train_np = np.zeros((len(trainInd), 2), dtype=int)
        for ind in range(len(trainInd)):
            train_np[ind] = [trainInd[ind][0], trainInd[ind][1]]
        self.train_np = train_np
        hard_dict = {}
        easy_dict = {}
        for t, (tshape, tcolor) in enumerate(train_np):
            easy_dict[t] = []
            hard_dict[t] = []
            for i, (shape, color) in enumerate(train_np):
                if i != t:
                    if shape != tshape and color != tcolor:
                        easy_dict[t].append(i)
                    else:
                        hard_dict[t].append(i)
        self.easy_dict = easy_dict
        self.hard_dict = hard_dict
        return train_np

    def getBatchData(self, indices, batch, distractNum, easy_limit = 5):

        # We need batch because we generate different instances consisting of all the distractors and target within
        # sample train batch from data
        # return numpy array [batch, attrLength]
        color = np.zeros([batch, distractNum, self.numColors], dtype=np.float32)
        shape = np.zeros([batch, distractNum, self.numShapes], dtype=np.float32)
        numTuples = len(indices) # number of tuples in the training set
        batchInd = [random.sample(range(numTuples), distractNum) for _ in range(batch)] # non-repetitive
        
        if easy_limit < distractNum - 1:
            targetInd = []
            for b, cand in enumerate(batchInd):
                easy_count = 0
                target = cand[0]
                candidates = cand[1:]
                for i in range(distractNum-1):
                    if candidates[i] in self.easy_dict[target]:
                        easy_count += 1
                        if easy_count > easy_limit:
                            candidates[i] = np.random.choice([x for x in self.hard_dict[target] if x not in candidates])
                candidates.append(target)
                random.shuffle(candidates)
                targetInd.append(candidates.index(target))
                batchInd[b] = candidates
        
        
        # fetch the batchid and turn color/shape index into one hot nunpy vertor
        for i in range(batch):
            for j in range(distractNum):
                colorindex = indices[batchInd[i][j]][0]
                shapeindex = indices[batchInd[i][j]][1]
                color[i][j][colorindex] = 1
                shape[i][j][shapeindex] = 1
        if self.attrSize != self.numColors + self.numShapes:
            x_coordinate = np.random.rand(batch, distractNum, 1)
            y_coordinate = np.random.rand(batch, distractNum, 1)
            instances = np.concatenate([color, shape, x_coordinate, y_coordinate], axis=2)
        else:
            instances = np.concatenate([color, shape], axis=2)
            
        if easy_limit >= distractNum - 1:
            # extract one target from distract tuples
            targetInd = np.random.randint(distractNum, size=(batch), dtype=int)
        targets = instances[np.arange(batch), targetInd, :]
        return instances, targets #(batch, distract, attrSize) (batch, attrSize)

    def getEnumerateData(self):
        attrVector = np.zeros([self.numColors * self.numShapes, self.numColors + self.numShapes], dtype=np.float32)
        for i in range(self.numColors):
            for j in range(self.numShapes):
                attrVector[i * self.numShapes + j][i] = 1
                attrVector[i * self.numShapes + j][self.numColors + j] = 1
        return attrVector
