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
    res = []
    newdata = np.zeros((w, h))
    tick = 0
    r = data.copy()
    while tick != TICK:
        z1, z2, r = splitdata(r)
        res.append(z1)
        res.append(z2)
        tick += 1
    for idx, d in enumerate(res):
        # print "scale of 0-1 matrix: {}/{}, 1 rate:{}".format(1, 2**(idx/2+1),
            # sum(d[d==1])/d.size)
        if idx % 2 == 0:
            newdata += (1.0 / 2**(idx / 2 + 1)) * d
        else:
            newdata -= (1.0 / 2**(idx / 2 + 1)) * d
    # print ("有 {} 个01矩阵".format(len(res)))
    return newdata


def main():
    data = np.loadtxt("greydata.txt", delimiter=",")
    data01, resdata = getres(data / 255)
    # cv2.imwrite("./pic/grey.png", data)
    # cv2.imwrite("./pic/zero-one.png", data01 * 255)
    # cv2.imwrite("./pic/residualdata.png", resdata * 255)
    for TICK in range(8):
        newdata = recmatrix(resdata, TICK=TICK)
        # print "误差最大是: {}".format(np.max(np.abs(newdata - resdata)))
        cv2.imwrite("./pic/{}_rec.png".format(TICK), (newdata + data01) * 255)
        cv2.imwrite("./pic/{}_diff.png".format(TICK), (newdata + data01) * 255 - data)
if __name__ == '__main__':
    main()
