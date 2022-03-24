# 読み込むときに輝度の値+RGBでフィルタリングすれば、テンプレートマッチングでも結構良い感じに予測できそう。
# 結局は過去に行っていたRGB輝度フィルタリングの矩形近似部分をテンプレートマッチングに置き換える感じ。

import cv2
import numpy


def main():
    # load the image from lists or folder
    list_name = "images_for_train"

    color = "red"
    # color = "blue"
    preprocess(color)

    return 0


def preprocess(color):
    # Filtering
    # Brightness Filtering : Amplitude of Gray-scale pixel value

    # RGB Filtering
    # it use BGR color profile
    if color == "blue":
        color_mask = (0, 0, 1)
    if color == "red":
        color_mask = (1, 0, 0)

    # Integration of these two results


    return 0


def train(lists):
    image_lists = read_image_name(lists)
    return None


def test(lists):
    image_lists = read_image_name(lists)
    return None


def read_image_name():
    image_lists = []
    return image_lists


if __name__ == '__main__':
    main()
