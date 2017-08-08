# coding=utf-8
from DecisionTree import DecisionTree
import TreePlotter


def getDataFromFile(filename):
    fr = open(filename, 'r')
    dataSet = [item.strip().split('\t') for item in fr.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataSet, labels


if __name__ == '__main__':
    tree = DecisionTree()
    dataSet, labels = getDataFromFile('lenses.txt')
    # mTree = tree.createTree(dataSet, labels[:])   # 以ID3算法构建决策树
    # tree.storeTree(mTree, 'classifyStorage.txt')
    mTree = tree.getTree('classifyStorage.txt')
    print tree.classify(['pre', 'myope', 'no', 'reduced'], labels[:], mTree)
    # TreePlotter.createPlot(mTree) # 作树状图
