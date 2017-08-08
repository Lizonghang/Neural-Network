# coding=utf-8
import math
import operator


class DecisionTree:

    """
        开始处理数据集时,首先需要测量集合中数据的信息熵,然后寻找最优方案划分数据集,直到数据集中的所有数据属于同一分类.
    ID3算法可以用于划分标称型数据集.构建决策树时,采用递归方法将数据集转化为决策树.其他决策树的构造算法最流行的是C4.5和
    CART.决策树分类算法是结果确定的算法,数据实例最终会被明确划分到某个分类中.
        优点: 计算复杂度不高,输出结果易于理解,对中间值的缺失不敏感,可以处理不相关特征数据.
        缺点: 可能会产生过度匹配的问题.
        适用数据: 数值型和标称型.
    """

    def __init__(self):
        pass

    def calcShannonEnt(self, dataSet):

        """
        :param dataSet:
        :return: 熵值shannonEnt,熵越高表明混合的数据越多,不确定度越大
        :description: 计算数据集的香农熵
        """

        numEntries = len(dataSet)
        labelCounts = {}
        for featVect in dataSet:
            currentLabel = featVect[-1]
            labelCounts[currentLabel] = labelCounts.get(currentLabel, 0.0) + 1
        shannonEnt = 0.0
        for label in labelCounts:
            prop = labelCounts[label] / numEntries
            shannonEnt -= prop * math.log(prop, 2)
        return shannonEnt

    def splitDataSet(self, dataSet, axis, value):

        """
        :param dataSet:
        :param axis: 以index=axis项作为特征
        :param value: 筛选出特征值满足该值时的特征向量
        :return: 特征满足特征值的其余特征向量组成的数据集
        :description: 划分数据集
        """

        retDataSet = []
        for featVect in dataSet:
            if featVect[axis] == value:
                reducedFeatVect = featVect[:axis]
                reducedFeatVect.extend(featVect[axis+1:])
                retDataSet.append(reducedFeatVect)
        return retDataSet

    def selectBestFeature(self, dataSet):

        """
        :param dataSet:
        :return: 选择最佳划分的特征索引
        :description: 选择最好的数据集划分方式
        """

        # 数据集初始香农熵
        baseEntropy = self.calcShannonEnt(dataSet)
        # 获取特征数量
        numFeatures = len(dataSet[0]) - 1
        # 记录最优信息增益和特征索引
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeatures):
            # 获取index=i列特征列表
            featList = [feats[i] for feats in dataSet]
            # 获取特征value种类
            values = set(featList)
            # 记录依据该特征分类后的香农熵
            currentEntropy = 0.0
            # 计算分类后的香农熵
            for value in values:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prop = len(subDataSet) / float(len(dataSet))
                currentEntropy += prop * self.calcShannonEnt(subDataSet)
            # 计算这种分类的信息增益
            infoGain = baseEntropy - currentEntropy
            # 获取最大信息增益对应的特征
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        # 返回最佳划分特征索引
        return bestFeature

    def majorityCnt(self, classList):

        """
        :param classList: 叶子节点类列表
        :return: 投票表决决定该叶子的分类
        :description: 如果数据集已经处理了所有属性,但类标签仍然不是唯一的,采用多数表决方式决定该叶子节点的分类
        """

        classCount = {}
        for vote in classList:
            classCount[vote] = classCount.get(vote, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):

        """
        :param dataSet:
        :param labels: 算法本身不需要该变量,它给出数据的明确意义
        :return: 字典存储的树信息mTree
        """

        # 获取全部类标签
        classList = [feats[-1] for feats in dataSet]
        # 类别完全相同,遍历终止
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 处理完所有的属性,依据多数表决方式决定该叶子节点分类
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        # 未满足终止条件,继续划分,选择最佳划分特征
        bestFeat = self.selectBestFeature(dataSet)
        # 该特征对应的特征名称
        bestFeatLabel = labels[bestFeat]
        # 初始化决策树,根节点为bestFeatLabel
        mTree = {bestFeatLabel: {}}
        # bestFeatLabel特征已使用,去除该特征
        del labels[bestFeat]
        # 获取该特征包含的所有可能值
        featValues = set([feats[bestFeat] for feats in dataSet])
        # 对每个值创建其分支
        for value in featValues:
            # 创建副本,若使用subLabels=labels传递的是列表的引用
            subLables = labels[:]
            # 递归创建子树
            mTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLables)
        # 返回决策树
        return mTree

    def classify(self, vect, labels, tree):

        """
        :param vect: 测试向量
        :param labels: 标签向量
        :param tree: 决策树
        :return: 决策结果classLabel
        """

        # 获取树的根节点标签
        rootLabel = tree.keys()[0]
        # 获取根节点的所有分支
        childBranches = tree[rootLabel]
        # 获取rootLabel对应的特征索引
        featIndex = labels.index(rootLabel)
        # 遍历所有分支对应的特征值情况,寻找适合的分支进入
        classLabel = ''
        for branch in childBranches.keys():
            if branch == vect[featIndex]:
                if type(childBranches[branch]).__name__ == 'dict':
                    classLabel = self.classify(vect, labels, childBranches[branch])
                else:
                    classLabel = childBranches[branch]
        return classLabel

    def storeTree(self, inputTree, filename):

        """
        :param inputTree: 存储的决策树
        :param filename: 存储文件名
        :return: None
        """

        import pickle
        fw = open(filename, 'w')
        pickle.dump(inputTree, fw)
        fw.close()

    def getTree(self, filename):

        """
        :param filename: 读取决策树的文件名
        :return: 决策树outputTree
        """

        import pickle
        fr = open(filename, 'r')
        outputTree = pickle.load(fr)
        fr.close()
        return outputTree

if __name__ == '__main__':
    tree = DecisionTree()
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    print u'计算dataSet的香农熵: ', tree.calcShannonEnt(dataSet)
    print u'按照第二个特征分割出特征值为1的数据集: ', tree.splitDataSet(dataSet, 1, 0)
    print u'从dataSet中选出最佳分割特征: ', tree.selectBestFeature(dataSet)
    print u'创建决策树: ', tree.createTree(dataSet, labels[:])
    mTree = tree.createTree(dataSet, labels[:])
    tree.storeTree(mTree, 'classifyStorage.txt')
    mTree = tree.getTree('classifyStorage.txt')
    print u'对向量[1,1]进行决策: ', tree.classify([1, 1], labels[:], mTree)
