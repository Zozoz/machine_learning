#!/usr/bin/env python
# encoding: utf-8


class treeNode:
    def __init__(self, length, depth, data, parentNode):
        self.length = length
        self.depth = depth
        self.dataList = data
        self.parent = parentNode
        self.lchild = None
        self.rchild = None

    def disp(self, ind=10):
        print ' '*ind, self.dataList
        if self.lchild != None:
            print '/', ' '*ind*2, '\\',
            self.lchild.disp(ind/2)
        if self.rchild != None:
            self.rchild.disp(ind/2)


def getData():
    return [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]

def initData(dataSet):
    dataMap = {}
    if dataSet == [] or dataSet == None:
        return [], {}
    for i in range(len(dataSet[0])):
        for j in range(len(dataSet)):
            if dataMap.has_key(i):
                dataMap[i].append(dataSet[j][i])
            else:
                dataMap[i] = [dataSet[j][i]]
    for i in dataMap.keys():
        dataMap[i].sort()
    return dataSet, dataMap

def createkdTree(dataSet, dataMap, parentNode, depth, k):
    length = len(dataSet)
    if length == 0:
        return None
    nowX = depth % k
    midValue = dataMap[nowX][length/2]
    print dataMap
    print 'midValue = ', midValue
    dataSetL = []
    dataSetR = []
    dataSetN = []
    for i in range(len(dataSet)):
        if dataSet[i][nowX] < midValue:
            dataSetL.append(dataSet[i])
        elif dataSet[i][nowX] > midValue:
            dataSetR.append(dataSet[i])
        else:
            dataSetN.append(dataSet[i])
    retTree = treeNode(length, depth, dataSetN, parentNode)
    print dataSetL
    dataSetL, dataMapL = initData(dataSetL)
    retTree.lchild = createkdTree(dataSetL, dataMapL, retTree, depth+1, k)
    dataSetR, dataMapR = initData(dataSetR)
    retTree.rchild = createkdTree(dataSetR, dataMapR, retTree, depth+1, k)
    return retTree

if __name__ == '__main__':
    dataSet, dataMap = initData(getData())
    retTree = createkdTree(dataSet, dataMap, None, 0, 2)
    retTree.disp()


