# coding=utf-8
import numpy
import operator


class kNN:

    """
    优点: 精度高,对异常值不敏感,无数据输入假定.
    缺点: 计算复杂度高,空间复杂度高.
        1.k近邻算法是基于实例的学习,适用算法必须有接近实际数据的训练样本数据.必须保存全部数据集,如果训练数据集很大,需要占用大量的存储空间.
        2.由于必须对数据集中每个数据计算距离值,实际使用非常耗时.
        3.无法给出数据的基础结构信息,因此无法判断平均实例样本和典型样本的特征.
    适用数据: 数值型和标称型
    """

    def __init__(self):
        pass

    def classify(self, inX, k, dataSet, labels):

        """
        :param inX: 需要判断分类的输入向量, 类型: list
        :param k: 近似点前k个点, 类型: int
        :param dataSet: 训练样本集, 类型: numpy.ndarray
        :param labels: 标签向量, 类型: list
        :return: label, 算法分类结果
        :description:
            k近似算法, KNN算法:
            1. 计算已知类别数据集中的点与当前点之间的距离
            2. 按照距离递增次序排序
            3. 选取与当前点距离最小的k个点
            4. 确定前k个点所在类别的出现频率
            5. 返回前k个点出现频率最高的类别作为当前点的预测分类
        :note: 由于涉及数学运算,训练集中的属性必须为数学运算符可识别类型
        """

        # 输入判断,dataSet和labels维度必须一致, k不能超过二者维数
        if dataSet.shape[0] != len(labels) or k > dataSet.shape[0]:
            print u'出现参数错误,dataSet和labels维度必须一致, k不能超过二者维数'
            return False
        # 根据欧拉距离公式计算点间距离
        dataSetSize = dataSet.shape[0]  # dataSet中点数量
        diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet   # inX作竖向扩展至与dataSet维度相同,并差操作
        sqDiffMat = diffMat ** 2    # 平方操作
        sqDistances = numpy.sum(sqDiffMat, 1)   # 横向求和
        distances = sqDistances ** 0.5  # 求根,得到inX与dataSet各点的点间距数组
        # 按距离递增排序,按列排序
        # argsort返回的是其值从小到大排列时对应的索引值,如:
        # distances = [2, 1, 3]
        # distances.argsort(0) 返回 [1, 0, 2]
        sortedDistIndicies = distances.argsort(0)
        # 选取前k个近似点,统计类别出现频率
        classCount = {}
        for i in range(k):
            label = labels[sortedDistIndicies[i]]
            classCount[label] = classCount.get(label, 0) + 1
        # 对前k个点的类别出现频率进行递减排序
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        # 返回前k个近似点中出现频率最高的类别作为预测分类
        return sortedClassCount[0][0]

    def getDataSetFromFile(self, filename):

        """
        :param filename: 训练数据所在文件, 要求每条数据独占一行, 行内以空白符分隔, 末尾列为类标签, 每行列数必须相同
        :return: 格式化的数据训练集retMat和类标签向量labels
        """

        # 读取文件数据
        with open(filename, 'r') as fr:
            arrayOfLines = fr.readlines()
        # 获取数据条数与每条数据特征数
        row = len(arrayOfLines)
        col = len(arrayOfLines[0].split()) - 1
        # 初始化数据集与分类集
        retMat = numpy.zeros((row, col))
        labels = []
        # 逐行读取特征与分类
        index = 0
        for line in arrayOfLines:
            items = line.split()
            retMat[index, :] = items[0: col]
            labels.append(items[-1])
            index += 1
        # 返回数据训练集retMat和类标签向量labels
        return retMat, labels

    def autoNorm(self, dataSet):

        """
        :param dataSet: 数据训练集dataSet
        :return: 归一化后的数据训练集dataSet
        :note: 准备数据,需要归一化数值,归一化公式: newValue = (oldValue - min) / (max - min)
        """

        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        row = dataSet.shape[0]
        normDataSet = (dataSet - numpy.tile(minVals, (row, 1))) / numpy.tile(ranges, (row, 1))
        return normDataSet, minVals, maxVals

    def getKNNPrecision(self, trainFilename, testFilename, k):

        """
        :param trainFilename: 训练集数据文件
        :param testFilename: 测试集数据文件
        :param k: 选取的近似点个数
        :return: 分类精度precision, 该值理论上不为1.0
        """

        # 训练集
        dataSet, labels = self.getDataSetFromFile(trainFilename)
        normDataSet, minVals, maxVals = self.autoNorm(dataSet)
        # 测试集
        testDataSet, testLabels = self.getDataSetFromFile(testFilename)
        testNormDataSet, testMinVals, testMaxVals = self.autoNorm(testDataSet)
        # 测试数据总数
        totalTestNum = testDataSet.shape[0]
        # 错误计数器
        errorCount = 0.0
        for i in range(totalTestNum):
            if testLabels[i] != self.classify(testNormDataSet[i, :], k, normDataSet, labels):
                errorCount += 1
        # 精度计算
        precision = 1 - errorCount / totalTestNum
        return precision


if __name__ == '__main__':
    knn = kNN()
    print u'算法精度: ', knn.getKNNPrecision('datingSet.txt', 'datingTestSet.txt', 10)
