import cv2
import numpy as np
from math import acos, pi, sqrt, pow


def getPrewittEdgeImg(threshold: int):
    src = cv2.imread('images/crowd.bmp', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    maskSize = 3

    maskX = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    maskY = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]

    # result img
    imgX = np.zeros((height, width, 1), dtype=np.uint8)
    imgY = np.zeros((height, width, 1), dtype=np.uint8)
    magnitudeImg = np.zeros((height, width, 1), dtype=np.uint8)
    resultImg = np.zeros((height, width, 1), dtype=np.uint8)

    # maskX 적용
    for y in range(height):
        for x in range(width):
            SUM1 = 0
            SUM2 = 0
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
                    SUM1 += src[new_y][new_x] * maskX[maskSize // 2 + i][maskSize // 2 + j]
                    SUM2 += src[new_y][new_x] * maskY[maskSize // 2 + i][maskSize // 2 + j]

            SUM1 = np.clip(SUM1 // (maskSize * maskSize), 0, 255)
            SUM2 = np.clip(SUM2 // (maskSize * maskSize), 0, 255)
            imgX[y][x] = SUM1
            imgY[y][x] = SUM2
            magnitudeImg[y][x] = pow(SUM1 * SUM1 + SUM2 * SUM2, 1/2)
            if magnitudeImg[y][x] > threshold:
                resultImg[y][x] = 255
            else:
                resultImg[y][x] = 0

    cv2.imshow('src', src)
    cv2.imshow('imgX', imgX)
    cv2.imshow('imgY', imgY)
    cv2.imshow('magnitudeImg', magnitudeImg)
    cv2.imshow('resultImg', resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getPrewittEdgeImg(8)


def getSobelEdgeImg(threshold: int):
    src = cv2.imread('images/man.bmp', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    maskSize = 3

    maskX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    maskY = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    # result img
    imgX = np.zeros((height, width, 1), dtype=np.uint8)
    imgY = np.zeros((height, width, 1), dtype=np.uint8)
    magnitudeImg = np.zeros((height, width, 1), dtype=np.uint8)
    resultImg = np.zeros((height, width, 1), dtype=np.uint8)

    # maskX 적용
    for y in range(height):
        for x in range(width):
            SUM1 = 0
            SUM2 = 0
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
                    SUM1 += src[new_y][new_x] * maskX[maskSize // 2 + i][maskSize // 2 + j]
                    SUM2 += src[new_y][new_x] * maskY[maskSize // 2 + i][maskSize // 2 + j]

            SUM1 = np.clip(SUM1 // (maskSize * maskSize), 0, 255)
            SUM2 = np.clip(SUM2 // (maskSize * maskSize), 0, 255)
            imgX[y][x] = SUM1
            imgY[y][x] = SUM2
            magnitudeImg[y][x] = pow(SUM1 * SUM1 + SUM2 * SUM2, 1/2)
            if magnitudeImg[y][x] > threshold:
                resultImg[y][x] = 255
            else:
                resultImg[y][x] = 0

    cv2.imshow('src', src)
    cv2.imshow('imgX', imgX)
    cv2.imshow('imgY', imgY)
    cv2.imshow('magnitudeImg', magnitudeImg)
    cv2.imshow('resultImg', resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getSobelEdgeImg(8)


def getLoGEdgeImg(maskSize: int, sigma, threshold):
    src = cv2.imread('images/noisy image/Fig0504(a)(gaussian-noise).jpg', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    # 마스크 사이즈가 1보다 작으면 리턴, 짝수면 홀수로 바꿔주기.
    if maskSize < 1:
        raise RuntimeError("maskSize can't be less than 1")
    if maskSize % 2 == 0:
        maskSize -= 1

    # 가우시안 필터 적용
    Gaussian_blur = cv2.GaussianBlur(src, (maskSize, maskSize), sigma)

    cv2.imshow('src', src)
    cv2.imshow('Gaussian_blur', Gaussian_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 라플라시안 연산자 적용
    laplacian = cv2.Laplacian(Gaussian_blur, cv2.CV_8U, maskSize)

    cv2.imshow('src', src)
    cv2.imshow('laplacian', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # thresholding
    for y in range(height):
        for x in range(width):
            if laplacian[y][x] > threshold:
                laplacian[y][x] = 255
            else:
                laplacian[y][x] = 0

    cv2.imshow('src', src)
    cv2.imshow('thresholded', laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# getLoGEdgeImg(5, 0.5, 8)


def getCannyEdgeImg(minVal: int, maxVal: int, maskSize, sigma):
    src = cv2.imread('images/안현재.jpeg', cv2.IMREAD_GRAYSCALE)
    height, width = src.shape[0], src.shape[1]

    # 가우시안 필터 적용
    Gaussian_blur = cv2.GaussianBlur(src, (maskSize, maskSize), sigma)

    # 캐니 연산자 적용
    result = cv2.Canny(Gaussian_blur, minVal, maxVal)

    cv2.imshow('src', src)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


getCannyEdgeImg(40, 80, 3, 1)