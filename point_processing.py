import cv2
import numpy as np
from math import acos, pi, sqrt, pow
import matplotlib.pyplot as plt


# B, G, R
def compute_Hue(B, G, R):
    angle = 0
    if B != G != R:
        angle = 0.5*((R-G)+(R-B)) / sqrt((R-G)*(R-G) + (R-B)*(G-B))
    return acos(angle) if B <= G else (2*pi - acos(angle))


def getSkinColor():
    src = cv2.imread('images/people.jpg', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]

    I = np.zeros((height, width))
    S = np.zeros((height, width))
    H = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            B, G, R = src[i][j][0] / 255., src[i][j][1] / 255., src[i][j][2] / 255.

            I[i][j] = (B + G + R) / 3.

            if B + G + R != 0:
                S[i][j] = 1 - 3 * np.min([B, G, R]) / (B + G + R)

            H[i][j] = compute_Hue(B, G, R)

    dst = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if 0.1 <= H[i][j] < 0.5 and 0.4 <= I[i][j] < 1:
                dst[i][j] = src[i][j]

    cv2.imshow('dst', dst)
    # cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getNegativeGrayScaleImg():
    src = cv2.imread('images/man.bmp', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    dst = np.zeros((height, width, 1), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            dst[i][j] = 255 - src[i][j]

    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getNegativeColorImg():
    src = cv2.imread('images/안현재.jpeg', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]

    dst = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            dst[i][j][0] = 255 - src[i][j][0]
            dst[i][j][1] = 255 - src[i][j][1]
            dst[i][j][2] = 255 - src[i][j][2]
    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getArithGrayscaleImg():
    src = cv2.imread('images/crowd.bmp', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    dst = np.zeros((height, width, 1), dtype=np.uint8)

    gamma = 0.6

    for i in range(height):
        for j in range(width):
            dst[i][j] = src[i][j] + 50
            if src[i][j] > 205:
                dst[i][j] = 255

    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getPowerLawGrayScaleImg(gamma):
    src = cv2.imread('images/crowd.bmp', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    dst = np.zeros((height, width, 1), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            dst[i][j] = 255 * pow(src[i][j]/255, gamma)

    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getPowerLawHueChangedColorImg(gamma):
    src = cv2.imread('images/lenna_color.bmp', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]
    dst = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            dst[i][j][0] = 255 * pow(src[i][j][0]/255, gamma)
            dst[i][j][1] = 255 * pow(src[i][j][1]/255, gamma)
            dst[i][j][2] = 255 * pow(src[i][j][2]/255, gamma)

    cv2.imshow('dst', dst)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def RGBToHSI(rgb):
    b = rgb[0]
    g = rgb[1]
    r = rgb[2]
    # I는 rgb값의 평균
    I = np.mean(rgb)

    if r == g == b:
        H = 0
        S = 0
    else:
        minRgb = min(rgb)
        S = 1 - (minRgb / I)

        temp = ((r - g) + (r - b)) / (2 * np.sqrt((r - g) * (r - g) + (r - b) * (g - b)))
        H = np.arccos(temp) * 180 / np.pi

        if b > g:
            H = 360 - H
        H /= 360

    return np.array([H, S, I], dtype=np.float32)


def HSIToRGB(hsi):
    h = hsi[0]
    s = hsi[1]
    i = hsi[2]
    # if black
    if i == 0:
        R = G = B = i
        return np.array([B, G, R], np.float32)
    # if grayscale
    if s == 0.0:
        R = G = B = i
        return np.array([B, G, R], np.float32)

    h *= 360
    if h <= 120:
        B = i * (1 - s)
        R = i * (1 + s * np.cos(h * np.pi / 180) / np.cos((60 - h) * np.pi / 180))
        G = 3 * i - (R + B)
    elif h <= 240:
        h -= 120
        R = i * (1 - s)
        G = i * (1 + s * np.cos(h * np.pi / 180) / np.cos((60 - h) * np.pi / 180))
        B = 3 * i - (R + G)
    else:
        h -= 240
        G = i * (1 - s)
        B = i * (1 + s * np.cos(h * np.pi / 180) / np.cos((60 - h) * np.pi / 180))
        R = 3 * i - (G + B)

    return np.array([B, G, R], dtype=np.float32)


def getPowerLawHueUnchangedColorImg(gamma):
    src = cv2.imread('images/lenna_color.bmp', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]

    imgTmp = src/255
    hsiImg = np.zeros(np.shape(src), dtype=np.float32)
    # RGB to HSI
    for i in range(height):
        for j in range(width):
            hsiImg[i, j] = RGBToHSI(imgTmp[i, j])

    # I값에 power law 적용
    for i in range(height):
        for j in range(width):
            hsiImg[i][j][2] = pow(hsiImg[i][j][2], gamma)

    # 다시 RGB로 변환
    newSrc = np.zeros(np.shape(src), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            newSrc[i, j] = HSIToRGB(hsiImg[i, j])

    newSrc = np.uint8(np.clip(newSrc*255, 0, 255))

    cv2.imshow('dst', newSrc)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getHEImg():
    src = cv2.imread('images/HE/Fig0316(1)(top_left).jpg', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    histogram = [0]*256
    histogramForShow = []

    # 히스토그램 계산, 히스토그램 표시
    for i in range(height):
        for j in range(width):
            histogram[src[i][j]] += 1
            histogramForShow.append(src[i][j])

    plt.hist(histogramForShow, bins=range(0, 255))
    plt.show()

    # get sum histogram
    sumOfHistogram = 0
    sumHistogram = [0] * 256
    scaleFactor = 255 / sum(histogram)
    for i in range(256):
        sumOfHistogram += histogram[i]
        sumHistogram[i] = round(sumOfHistogram * scaleFactor)

    dst = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            dst[i][j] = sumHistogram[src[i][j]]

    newHistogramForShow = []

    # 새 히스토그램 계산, 히스토그램 표시
    for i in range(height):
        for j in range(width):
            newHistogramForShow.append(dst[i][j][0])

    plt.hist(newHistogramForShow, bins=range(0, 255))
    plt.show()

    cv2.imshow('result', dst)
    cv2.imshow('origin', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getColorHEImg():
    src = cv2.imread('images/안현재.jpeg', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]

    histogram1 = [0]*256

    # 히스토그램 계산, 히스토그램 표시
    for i in range(height):
        for j in range(width):
            histogram1[src[i][j][0]] += 1


    histogram2 = [0] * 256

    # 히스토그램 계산, 히스토그램 표시
    for i in range(height):
        for j in range(width):
            histogram2[src[i][j][1]] += 1

    histogram3 = [0] * 256

    # 히스토그램 계산, 히스토그램 표시
    for i in range(height):
        for j in range(width):
            histogram3[src[i][j][2]] += 1

    # 채널 별 평활화
    dst = np.zeros((height, width, 3), dtype=np.uint8)
    sumOfHistogram = 0
    sumHistogram = [0] * 256
    scaleFactor = 255 / sum(histogram1)
    for i in range(256):
        sumOfHistogram += histogram1[i]
        sumHistogram[i] = round(sumOfHistogram * scaleFactor)

    for i in range(height):
        for j in range(width):
            dst[i][j][0] = sumHistogram[src[i][j][0]]

    sumOfHistogram = 0
    sumHistogram = [0] * 256
    for i in range(256):
        sumOfHistogram += histogram2[i]
        sumHistogram[i] = round(sumOfHistogram * scaleFactor)

    for i in range(height):
        for j in range(width):
            dst[i][j][1] = sumHistogram[src[i][j][1]]

    sumOfHistogram = 0
    sumHistogram = [0] * 256
    for i in range(256):
        sumOfHistogram += histogram3[i]
        sumHistogram[i] = round(sumOfHistogram * scaleFactor)

    for i in range(height):
        for j in range(width):
            dst[i][j][2] = sumHistogram[src[i][j][2]]

    cv2.imshow('result', dst)
    cv2.imshow('origin', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()