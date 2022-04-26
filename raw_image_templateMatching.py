# 読み込むときに輝度の値+RGBでフィルタリングすれば、テンプレートマッチングでも結構良い感じに予測できそう。
# 結局は過去に行っていたRGB輝度フィルタリングの矩形近似部分をテンプレートマッチングに置き換える感じ。
# ここではとりあえず普通に画像に対してテンプレートマッチングしてみる
import cv2
import sys
import copy
import numpy as np

def main():
    # 入力画像とテンプレート画像をで取得
    # 入力画像とテンプレート画像をで取得
    folder_name = "./images_darked/"
    img = cv2.imread(folder_name + "17.jpg")
    temp = cv2.imread(folder_name + "target1.jpg")

    print(img.shape)
    print(temp.shape)

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # テンプレート画像の高さ・幅
    h, w = temp.shape

    # テンプレートマッチング（OpenCVで実装）SSD(Sum of Absolute Difference) 画素値の差分の二乗値の和
    # match = cv2.matchTemplate(gray, temp, cv2.TM_SQDIFF)
    # min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    # pt = min_pt

    # テンプレートマッチング（OpenCVで実装）ZNCC（Zero-mean Normalized Cross Correlation）ゼロ平均正規化相互相関
    match = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = max_pt

    # テンプレートマッチングの結果を出力
    cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)

    # 画像表示
    cv2.imshow('img', img)

    # キー押下で終了
    cv2.waitKey(0)

    return 0


if __name__ == '__main__':
    main()
