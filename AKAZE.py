import numpy as np
import cv2
import sys
import copy

def main():
    # 入力画像とテンプレート画像をで取得
    folder_name = "./images_darked/"
    img = cv2.imread(folder_name + "8.jpg")
    temp = cv2.imread(folder_name + "target1.jpg")

    print(searchPosition(temp,img))

def searchPosition(img1,img2, good_match_rate=0.30):
    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # 特徴量のマッチングを実行
    bf = cv2.BFMatcher()  # 総当たりマッチング(Brute-Force Matcher)生成
    matches = bf.knnMatch(des1, des2, k=2)

    # データをマッチング精度の高いもののみ抽出
    ratio = 0.5
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    # 対応する特徴点同士を描画
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    # 画像表示
    cv2.imshow('img', img3)

    # キー押下で終了
    cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    main()