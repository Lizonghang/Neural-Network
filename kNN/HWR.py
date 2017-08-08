# coding=utf-8
from kNN import kNN
import numpy
import os


# 将文本中的图像矩阵转换为向量
def imgTxt2Vector(filename):
    with open(filename, 'r') as fr:
        pixelVect = fr.readlines()
    row = len(pixelVect)
    col = len(pixelVect[0].strip())
    retVect = numpy.zeros((1, row * col))
    for i in range(row):
        for j in range(col):
            retVect[0, i * col + j] = pixelVect[i].strip()[j]
    return retVect, row, col


# 手写数字识别算法测试, 分类精度0.978858350951
def HWRTest(trainingDir, testDir, k):
    knn = kNN()
    # 获取训练集数据
    trainingFileList = os.listdir(trainingDir)
    m = len(trainingFileList)
    imgVect, row, col = imgTxt2Vector('/'.join((trainingDir, trainingFileList[0])))
    trainingMat = numpy.zeros((m, row * col))
    hwLabel = []
    for i in range(m):
        hwLabel.append(trainingFileList[i].split('_')[0])
        trainingMat[i, :] = imgTxt2Vector('/'.join((trainingDir, trainingFileList[i])))[0]
    # 使用测试集数据测试算法精度
    testFileList = os.listdir(testDir)
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        classRealNum = testFileList[i].split('_')[0]
        testVector = imgTxt2Vector('/'.join((testDir, testFileList[i])))[0]
        # 文件中的值已经在0和1之间,不需要归一化
        classResultNum = knn.classify(testVector, k, trainingMat, hwLabel)
        if classRealNum != classResultNum:
            errorCount += 1
    precision = 1 - errorCount / mTest
    return precision


if __name__ == '__main__':
    # 测试分类精度
    print HWRTest('trainingDigits', 'testDigits', 10)
    # 测试识别单文件数值
    knn = kNN()
    # 获取训练集数据
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    imgVect, row, col = imgTxt2Vector('/'.join(('trainingDigits', trainingFileList[0])))
    trainingMat = numpy.zeros((m, row * col))
    hwLabel = []
    for i in range(m):
        hwLabel.append(trainingFileList[i].split('_')[0])
        trainingMat[i, :] = imgTxt2Vector('/'.join(('trainingDigits', trainingFileList[i])))[0]
    # 获取需要识别的文本数据
    imgVector = imgTxt2Vector('testDigits/5_5.txt')[0]
    # 识别操作
    print knn.classify(imgVector, 10, trainingMat, hwLabel)
