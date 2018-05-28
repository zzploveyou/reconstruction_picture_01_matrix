# coding:utf-8
import cv2
import numpy as np

THRESHOLD = 0.5

def getres(data):
    """
    Usage
    -----
    split normalized data to zero-one and residual

    Parameters
    ----------
    data: numpy.ndarray
        2-D data.

    Returns
    ----------
    data1_zero_one: numpy.ndarray
        01 matrix.
    res: numpy.ndarray
        residual part.

    """
    global THRESHOLD
    w, h = data.shape
    data1_zero_one = np.zeros((w, h))
    for (idx, idy), d in np.ndenumerate(data):
        if d >= THRESHOLD:
            data1_zero_one[idx, idy] = 1
    res = data - data1_zero_one
    return data1_zero_one, res


def splitdata(data):
    """
    Parameters
    ----------
    data: numpy.ndarray
        2-D data.

    Returns
    ----------
    data1_zero_one: numpy.ndarray
        positive part.
    data1_zero_one: numpy.ndarray
        negative part.
    data1_res-data2_res: numpy.ndarray
        residue data.

    """
    w, h = data.shape
    data1 = np.zeros((w, h))
    data2 = np.zeros((w, h))
    for (idx, idy), d in np.ndenumerate(data):
        if d > 0:
            data1[idx, idy] = d
        if d < 0:
            data2[idx, idy] = -d
    data1 = data1 * 2
    data1_zero_one, data1_res = getres(data1)
    data2 = data2 * 2
    data2_zero_one, data2_res = getres(data2)
    return data1_zero_one, data2_zero_one, data1_res - data2_res

def recmatrix(data, TICK):
    """将矩阵分解为一系列01矩阵，并通过01矩阵的加权和恢复原始矩阵"""
    w, h = data.shape
    Aks = []
    newdata = np.zeros((w, h))
    tick = 0
    r = data.copy()
    while tick != TICK:
        z1, z2, r = splitdata(r)
        Aks.append(z1)
        Aks.append(z2)
        tick += 1
    for idx in range(len(Aks)):
        if idx % 2 == 0:
            p = (1.0 / 2**(idx / 2 + 1))
            B = Aks[idx] - Aks[idx+1]
            newdata += p * B
            yield p, B, newdata

def main0():
    data = np.loadtxt("greydata.txt", delimiter=",", dtype=np.uint8)
    data01, resdata = getres(data / 255.0)
    cv2.imwrite("./pic/grey.png", data)
    cv2.imwrite("./pic/zero-one.png", data01 * 255)
    cv2.imwrite("./pic/residualdata.png", resdata * 255)
    idx = 1
    for p, B, newdata in recmatrix(resdata, TICK=10):
        # print "误差最大是: {}".format(np.max(np.abs(newdata - resdata)))
        cv2.imwrite("./pic/rec_{}.png".format(idx), (newdata + data01) * 255)
        cv2.imwrite("./pic/B_{}.png".format(idx), B * 255)
        cv2.imwrite("./pic/pB_{}.png".format(idx), B * 255 * p)
        cv2.imwrite("./pic/res_{}.png".format(idx+1), ((newdata + data01) * 255 - data)/p)
        cv2.imwrite("./pic/pres_{}.png".format(idx+1), (newdata + data01) * 255 - data)
        print (((newdata + data01) * 255 - data)/p)[0, 0]/255
        # print ((newdata + data01) * 255 - data).max()
        # print "{}: 恢复后数据与原始数据差异最大的元素的值:{}".format(
        #     idx, (cv2.imread("./pic/rec_{}.png".format(idx))[:, :, 0]-data).max())
        idx += 1

def recmatrix2(data):
    w, h = data.shape
    Cks = []
    newdata = np.zeros((w, h))
    for i in range(8):
        tmpdata = np.zeros((w, h))
        for (idx, idy), d in np.ndenumerate(data):
            tmpdata[idx, idy] = (d >> (7-i)) % 2
        tmpdata = 2**(7-i)*tmpdata
        newdata += tmpdata
        # yield tmpdata*2**(i+1), newdata
        yield tmpdata, newdata

def main1():
    data = np.loadtxt("greydata.txt", delimiter=",", dtype=np.uint8)
    idx = 1
    for tmpdata, newdata in recmatrix2(data):
        cv2.imwrite("./pic2/each0_{}.png".format(idx), tmpdata)
        cv2.imwrite("./pic2/rec_{}.png".format(idx), newdata)
        print("diff: {}".format(abs(newdata-data).max()))
        idx += 1
if __name__ == '__main__':
    # main0()
    main1()
