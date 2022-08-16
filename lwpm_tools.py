import cv2
import numpy as np
# 検出するhsvの範囲を設定
inrange_color = {"red0": [(0, 40, 80), (30, 255, 255)],
                 "red1": [(151, 40, 80), (179, 255, 255)],
                 "blue": [(90, 40, 150), (149, 255, 255)]}
# 画像の高さ大きさに対するkernel sizeの比(画像高さ:kernel_ratio * 2 + 1 )
kernel_ratio = 200
# 処理する輪郭の最大数
max_num_of_contours = 10
# カラーバーとして認識する最小の輪郭面積比
area_th = 0.00001
# カラーバーのbounding boxを長方形として認識する最小の比
# (長辺を縦と仮定したとき，縦/横)
aspect_th = 2
# 画像縦方向と，bounding box長辺の成す角の最大値
theta_th = np.pi / 4
# bestMatchするbounding box を探索する回数
max_num_of_searches = 20
# 2つの bounding box の長辺の長さの比の閾値
length_ratio_th = 0.7
# 2つの bounding box の長辺が成す角の閾値
parallel_th = np.pi * (0.06)
# 2つの bounding box の距離(中心距離)のスレッショルド
# 2つの box の長辺長さの平均を1として，boxの最小距離と最大距離を記述．
box_distance_th = [1, 3.5]
def inRangeRed(img):
    img_th1 = cv2.inRange(img, inrange_color["red0"][0], inrange_color["red0"][1])
    img_th2 = cv2.inRange(img, inrange_color["red1"][0], inrange_color["red1"][1])
    return img_th1 + img_th2
def inRangeBlue(img):
    return cv2.inRange(img, inrange_color["blue"][0], inrange_color["blue"][1])
inRange = {"red": inRangeRed, "blue": inRangeBlue}
# boundingboxの縦横比を返す(長辺を縦と仮定したとき，縦/横)
def aspectRatio(box):
    e1 = box[0] - box[1]
    e2 = box[1] - box[2]
    l1 = np.sqrt((e1 * e1).sum())
    l2 = np.sqrt((e2 * e2).sum())
    ma = max(l1, l2)
    mi = min(l1, l2)
    return ma / mi
# 画像鉛直方向と bounding box の長辺の角度を求める
def radianOnVertical(box):
    e1 = box[0] - box[1]
    e2 = box[1] - box[2]
    ma = max(e1, e2, key=lambda x: np.sqrt(x * x).sum())
    l = np.sqrt(ma * ma).sum()
    theta = np.arccos(ma[1] / l)
    return min(theta, np.pi - theta)
# bounding box の長辺の長さのみを返す
def longSideLength(box):
    e1 = box[0] - box[1]
    e2 = box[1] - box[2]
    l1 = np.sqrt((e1 * e1).sum())
    l2 = np.sqrt((e2 * e2).sum())
    ma = max(l1, l2)
    return ma
# bounding box の長辺ベクトルとその長さを返す
def longSide(box):
    e1 = box[0] - box[1]
    e2 = box[1] - box[2]
    e_max = max(e1, e2, key=lambda x: np.sqrt(x * x).sum())
    l_max = np.sqrt((e_max * e_max).sum())
    return e_max, l_max
# 装甲板に外接する矩形を計算
# return : 外接矩形の左上座標 , 幅-高さ
def calc_outbox(box1, box2):
    x_min = min(np.append(box1[:, 0], box2[:, 0]))
    y_min = min(np.append(box1[:, 1], box2[:, 1]))
    x_max = max(np.append(box1[:, 0], box2[:, 0]))
    y_max = max(np.append(box1[:, 1], box2[:, 1]))
    width = x_max - x_min
    height = y_max - y_min
    return np.array([x_min, y_min]), np.array([width, height])
# 最も長いカラーバーに対して平行かつ，長さが一定以上ある他のカラーバーを選択し，狙う座標を返す
def bestMatch(boxes):
    detected_list = []  # 条件に合うすべての装甲板を格納しreturnする
    if len(boxes) <= 1:
        return detected_list
    max_searches = min(max_num_of_searches, len(boxes) - 1)
    for base in range(max_searches):
        e_max, l_max = longSide(boxes[base])
        for i in range(base + 1, max_searches + 1):
            e, l = longSide(boxes[i])
            # 矩形領域の長辺の長さでフィルタリング
            length_ratio = l / l_max
            if length_ratio < length_ratio_th: # 2長辺の長さがかけな離れている
                # print("length_ratio", base, i)
                # print(length_ratio_th, length_ratio)
                break
            # 2つの矩形領域(2長辺)の平行性でフィルタリング
            theta = np.arccos((e * e_max).sum() / (l * l_max))
            theta = min(np.pi - theta, theta)
            if theta > parallel_th:
                # print("parallel_th ", base, i)
                # print(parallel_th, theta)
                continue
            dis = (np.sum(boxes[base], axis=0) - np.sum(boxes[i], axis=0)) / 4
            l_mean = (l_max + l) / 2
            dis = np.sqrt((dis * dis).sum()) / (l_mean)
            # box 間距離が範囲内にあるならboxの中心座標を出力
            if box_distance_th[0] < dis and dis < box_distance_th[1]:
                # print("return", base, i)
                center = np.sum((boxes[base] + boxes[i]), axis=0) // 8
                left_upper, wh = calc_outbox(boxes[base],boxes[i])
                # print(f"{boxes[base]} and {boxes[i]}")
                detected_list.append([center, wh, left_upper]) # 条件に合う装甲板をlistに格納
    return detected_list
# 標的座標を出力する
def determinate(img, enemy_color):
    kernel_size = (img.shape[0] // kernel_ratio) * 2 + 1
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img_show = img.copy()
    # hsvを利用して2値化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = inRange[enemy_color](img)
    img_th = img.copy()
    # クローズ (膨張->収縮)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 有力なカラーバーの bounding box を格納
    boxes = []
    # 外側の輪郭のみ抽出
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 有力なカラーバーを抽出
    img_area = img.shape[0] * img.shape[1]
    for i, cnt in enumerate(contours):
        # 輪郭面積による抽出
        area_ratio = cv2.contourArea(cnt) / img_area
        if area_ratio < area_th:
            # info += "area_ratio " + str(area_ratio) + "\n\n"
            continue
        # アスペクト比と，垂直方向と長辺の成す角による抽出
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if aspectRatio(box) < aspect_th or radianOnVertical(box) >= theta_th:
            continue
        # 抽出されたカラーバーのbounding box
        box = np.int0(box)
        boxes.append(box)
    # お好みの方法でソート．一番長辺の長さが長いbounding box を持つ装甲板が標的になるように処理される
    boxes.sort(key=lambda box: cv2.contourArea(box.reshape(-1, 1, 2)), reverse=True)
    # ターゲット座標を決定する
    print(f"{len(boxes)} Boxes")
    return bestMatch(boxes)
    # TODO: remove img_show, img_th
