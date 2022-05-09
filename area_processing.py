from point_processing import getHEImg, RGBToHSI, HSIToRGB

import cv2
import numpy as np
import random
from math import acos, pi, sqrt, pow


def getMeanFilteredGrayscaleImg(maskSize: int):
    src = cv2.imread('images/noisy image/Fig0504(i)(salt-pepper-noise).jpg', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    # mask 정의
    # 마스크 사이즈가 1보다 작으면 리턴, 짝수면 홀수로 바꿔주기.
    if maskSize < 1:
        raise RuntimeError("maskSize can't be less than 1")
    if maskSize % 2 == 0:
        maskSize -= 1
    mask = np.ones((maskSize, maskSize))

    # result img
    resultImg = np.zeros((height, width, 1), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            SUM = 0
            for i in range(-1 * maskSize // 2, maskSize // 2):
                for j in range(-1 * maskSize // 2, maskSize // 2):
                    new_y = y + i
                    new_x = x + j

                    # 영상 범위를 벗어난 경우 테두리 값을 사용
                    if new_y < 0:
                        new_y = 0
                    elif new_y > height - 1:
                        new_y = height - 1
                    if new_x < 0:
                        new_x = 0
                    elif new_x > width - 1:
                        new_x = width - 1

                    # 컨볼루션 계산
                    SUM += src[new_y][new_x] * mask[maskSize // 2 + i][maskSize // 2 + j]

            SUM = np.clip(SUM // (maskSize * maskSize), 0, 255)
            resultImg[y][x] = SUM

    cv2.imshow('src', src)
    cv2.imshow('result', resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getMeanFilteredGrayscaleImg(3)


def getMeanFilteredColorImg(maskSize: int):
    src = cv2.imread('images/Color noisy image/Salt&pepper noise.png', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]

    # 마스크 사이즈가 1보다 작으면 리턴, 짝수면 홀수로 바꿔주기.
    if maskSize < 1:
        raise RuntimeError("maskSize can't be less than 1")
    if maskSize % 2 == 0:
        maskSize -= 1
    # mask 정의
    mask = np.ones((maskSize, maskSize))

    imgTmp = src / 255
    hsiImg = np.zeros(np.shape(src), dtype=np.float32)

    # RGB to HSI
    for i in range(height):
        for j in range(width):
            hsiImg[i, j] = RGBToHSI(imgTmp[i, j])
    hsiImg *= 255
    # I 값에 MeanFilter 적용
    newHsiImg = hsiImg

    for y in range(height):
        for x in range(width):
            SUM = 0
            for i in range(-1 * maskSize // 2, maskSize // 2):
                for j in range(-1 * maskSize // 2, maskSize // 2):
                    new_y = y + i
                    new_x = x + j

                    # 영상 범위를 벗어난 경우 테두리 값을 사용
                    if new_y < 0:
                        new_y = 0
                    elif new_y > height - 1:
                        new_y = height - 1
                    if new_x < 0:
                        new_x = 0
                    elif new_x > width - 1:
                        new_x = width - 1

                    # 컨볼루션 계산
                    SUM += hsiImg[new_y][new_x][2] * mask[maskSize // 2 + i][maskSize // 2 + j]

            SUM = np.clip(SUM // (maskSize * maskSize), 0, 255)
            newHsiImg[y][x][2] = SUM // maskSize * maskSize

    newHsiImg /= 255

    # newHsiImg를 다시 RGB로 변환
    newSrc = np.zeros(np.shape(src), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            newSrc[i, j] = HSIToRGB(newHsiImg[i, j])

    newSrc = np.uint8(np.clip(newSrc * 255, 0, 255))

    cv2.imshow('src', src)
    cv2.imshow(str(maskSize), newSrc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getMeanFilteredColorImg(3)


def getGaussianFilteredGrayscaleImg(maskSize, sigma):
    src = cv2.imread('images/noisy image/Fig0504(a)(gaussian-noise).jpg', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    # 마스크 사이즈가 1보다 작으면 리턴, 짝수면 홀수로 바꿔주기.
    if maskSize < 1:
        raise RuntimeError("maskSize can't be less than 1")
    if maskSize % 2 == 0:
        maskSize -= 1

    # mask 정의
    mask = np.zeros((maskSize, maskSize))
    mean = maskSize // 2
    sumOfKernerValue = 0
    for x in range(maskSize):
        for y in range(maskSize):
            mask[x][y] = np.exp(-0.5 * (pow((x - mean) / sigma, 2) + pow((y - mean) / sigma, 2))) / (
                    2 * pi * sigma * sigma)
            sumOfKernerValue += mask[x][y]

    # 마스크 정규화
    for x in range(maskSize):
        for y in range(maskSize):
            mask[x][y] /= sumOfKernerValue

    # result img
    resultImg = np.zeros((height, width, 1), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            SUM = 0
            for i in range(-1 * maskSize // 2, maskSize // 2):
                for j in range(-1 * maskSize // 2, maskSize // 2):
                    new_y = y + i
                    new_x = x + j

                    # 영상 범위를 벗어난 경우 테두리 값을 사용
                    if new_y < 0:
                        new_y = 0
                    elif new_y > height - 1:
                        new_y = height - 1
                    if new_x < 0:
                        new_x = 0
                    elif new_x > width - 1:
                        new_x = width - 1

                    # 컨볼루션 계산
                    SUM += src[new_y][new_x] * mask[maskSize // 2 + i][maskSize // 2 + j]

            resultImg[y][x] = SUM

    cv2.imshow('src', src)
    cv2.imshow(str(maskSize), resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getGaussianFilteredGrayscaleImg(5, 2)


def getGaussianFilteredColorImg(maskSize, sigma):
    src = cv2.imread('images/Color noisy image/Salt&pepper noise.png', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]

    # 마스크 사이즈가 1보다 작으면 리턴, 짝수면 홀수로 바꿔주기.
    if maskSize < 1:
        raise RuntimeError("maskSize can't be less than 1")
    if maskSize % 2 == 0:
        maskSize -= 1
    # 마스크 정의
    mask = np.zeros((maskSize, maskSize))
    mean = maskSize // 2
    sumOfKernerValue = 0
    for x in range(maskSize):
        for y in range(maskSize):
            mask[x][y] = np.exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (
                    2 * pi * sigma * sigma)
            sumOfKernerValue += mask[x][y]

    # 마스크 정규화
    for x in range(maskSize):
        for y in range(maskSize):
            mask[x][y] /= sumOfKernerValue

    imgTmp = src / 255
    hsiImg = np.zeros(np.shape(src), dtype=np.float32)

    # RGB to HSI
    for i in range(height):
        for j in range(width):
            hsiImg[i, j] = RGBToHSI(imgTmp[i, j])
    hsiImg *= 255
    # I 값에 MeanFilter 적용
    newHsiImg = hsiImg

    for y in range(height):
        for x in range(width):
            SUM = 0
            for i in range(-1 * maskSize // 2, maskSize // 2):
                for j in range(-1 * maskSize // 2, maskSize // 2):
                    new_y = y + i
                    new_x = x + j

                    # 영상 범위를 벗어난 경우 테두리 값을 사용
                    if new_y < 0:
                        new_y = 0
                    elif new_y > height - 1:
                        new_y = height - 1
                    if new_x < 0:
                        new_x = 0
                    elif new_x > width - 1:
                        new_x = width - 1

                    # 컨볼루션 계산
                    SUM += hsiImg[new_y][new_x][2] * mask[maskSize // 2 + i][maskSize // 2 + j]

            newHsiImg[y][x][2] = SUM

    newHsiImg /= 255

    # newHsiImg를 다시 RGB로 변환
    newSrc = np.zeros(np.shape(src), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            newSrc[i, j] = HSIToRGB(newHsiImg[i, j])

    newSrc = np.uint8(np.clip(newSrc * 255, 0, 255))

    cv2.imshow('src', src)
    cv2.imshow(str(maskSize), newSrc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getGaussianFilteredColorImg(5, 2)


def getMedianFilteredGrayscaleImg(maskSize):
    src = cv2.imread('images/noisy image/Fig0504(a)(gaussian-noise).jpg', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    # 마스크 사이즈가 1보다 작으면 리턴, 짝수면 홀수로 바꿔주기.
    if maskSize < 1:
        raise RuntimeError("maskSize can't be less than 1")
    if maskSize % 2 == 0:
        maskSize -= 1

    # 마스크 안에 들어오는 값들을 저장, 정렬하여 중간값을 뽑기위한 배열
    buffer = []

    # result img
    resultImg = np.zeros((height, width, 1), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            median = 0
            for i in range(-1 * maskSize // 2, maskSize // 2):
                for j in range(-1 * maskSize // 2, maskSize // 2):
                    new_y = y + i
                    new_x = x + j

                    # 영상 범위를 벗어난 경우 테두리 값을 사용
                    if new_y < 0:
                        new_y = 0
                    elif new_y > height - 1:
                        new_y = height - 1
                    if new_x < 0:
                        new_x = 0
                    elif new_x > width - 1:
                        new_x = width - 1

                    # 컨볼루션 계산
                    buffer.append(src[new_y][new_x])
            # 정렬
            buffer.sort()
            # 중간값 뽑기
            median = buffer[maskSize * maskSize // 2]
            buffer.clear()
            resultImg[y][x] = median

    cv2.imshow('src', src)
    cv2.imshow(str(maskSize), resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getMedianFilteredGrayscaleImg(5)


def getHighBoostGrayscaleImg(a, maskType):
    src = cv2.imread('images/안현재.jpeg', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    # a = np.clip(a, 1, 2)

    maskSize = 3

    # mask 정의
    mask1 = [[0, -1, 0], [-1, a + 4, -1], [0, -1, 0]]
    mask2 = [[-1, -1, -1], [-1, a + 8, -1], [-1, -1, -1]]

    if maskType == 1:
        mask = mask1
    else:
        mask = mask2

    # result img
    resultImg = np.zeros((height, width, 1), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            SUM = 0
            for i in range(-1 * maskSize // 2 + 1, maskSize // 2+1):
                for j in range(-1 * maskSize // 2+1, maskSize // 2+1):
                    new_y = y + i
                    new_x = x + j

                    # 영상 범위를 벗어난 경우 테두리 값을 사용
                    if new_y < 0:
                        new_y = 0
                    elif new_y > height - 1:
                        new_y = height - 1
                    if new_x < 0:
                        new_x = 0
                    elif new_x > width - 1:
                        new_x = width - 1

                    # 컨볼루션 계산
                    SUM += src[new_y][new_x] * mask[maskSize // 2 + i][maskSize // 2 + j]
            SUM = np.clip(SUM, 0, 255)
            resultImg[y][x] = SUM

    cv2.imshow('src', src)
    cv2.imshow(str(a), resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getHighBoostGrayscaleImg(1.9, 2)


def getHighBoostColorImg(a, maskType):
    src = cv2.imread('images/High-boost filter images/Fig0327(a)(tungsten_original).jpg', cv2.IMREAD_COLOR)
    height, width = src.shape[0], src.shape[1]

    # a = np.clip(a, 1, 2)

    maskSize = 3

    # mask 정의
    mask1 = [[0, -1, 0], [-1, a + 4, -1], [0, -1, 0]]
    mask2 = [[-1, -1, -1], [-1, a + 8, -1], [-1, -1, -1]]

    if maskType == 1:
        mask = mask1
    else:
        mask = mask2

    imgTmp = src / 255
    hsiImg = np.zeros(np.shape(src), dtype=np.float32)

    # RGB to HSI
    for i in range(height):
        for j in range(width):
            hsiImg[i, j] = RGBToHSI(imgTmp[i, j])

    hsiImg *= 255
    newHsiImg = hsiImg

    for y in range(height):
        for x in range(width):
            SUM = 0
            for i in range(-1 * maskSize // 2 + 1, maskSize // 2 +1):
                for j in range(-1 * maskSize // 2+1, maskSize // 2+1):
                    new_y = y + i
                    new_x = x + j

                    # 영상 범위를 벗어난 경우 테두리 값을 사용
                    if new_y < 0:
                        new_y = 0
                    elif new_y > height - 1:
                        new_y = height - 1
                    if new_x < 0:
                        new_x = 0
                    elif new_x > width - 1:
                        new_x = width - 1

                    # 컨볼루션 계산
                    SUM += hsiImg[new_y][new_x][2] * mask[maskSize // 2 + i][maskSize // 2 + j]
            SUM = np.clip(SUM, 0, 255)
            newHsiImg[y][x][2] = SUM

    newHsiImg /= 255

    # newHsiImg를 다시 RGB로 변환
    newSrc = np.zeros(np.shape(src), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            newSrc[i, j] = HSIToRGB(newHsiImg[i, j])

    newSrc = np.uint8(np.clip(newSrc * 255, 0, 255))

    cv2.imshow('src', src)
    cv2.imshow(str(a), newSrc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_noise(noiseNumber):
    src = cv2.imread('images/noisy image/Fig0503 (original_pattern).jpg', cv2.IMREAD_GRAYSCALE)

    row, col = src.shape

    for i in range(noiseNumber):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)

        src[y_coord][x_coord] = 255

    for i in range(noiseNumber):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)

        src[y_coord][x_coord] = 0
    cv2.imwrite('images/noisy image/salpep' + str(noiseNumber) + ' .jpg', src)
    cv2.imshow('src', src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#
# add_noise(28000)
#
