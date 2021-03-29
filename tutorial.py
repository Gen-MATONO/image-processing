from skimage import io
import numpy as np
import cv2 # opencv
import matplotlib.pyplot as plt


img_original = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/assets/imori_256x256.png') # skimageを使ったwebからの画像読み込み
# opencvを使って既存フォルダから画像を読み込むと画像のデータ順序はRとBが反転しているため，正規の色で表現したいときは色を入れ替える必要あり
# io.imreadで読み込んだ画像をimg_original.dtypeで見てみるとuinitになっている．これは正負のない整数データであることを示す．

img = img_original.copy().astype(np.float64) # 元のデータを汚さないために.copy()でコピーを作り.astype(float64)で倍精度浮動小数にデータを変形している．
# 浮動小数型に変更する理由はデータをいじる際に整数だと切り捨て誤差や0, 255付近でデータが足りない，負の処理ができないなどの様々な不便が生じるから．

img /=255 # init型は[0, 255]で画像出力ができるがfloat型を画像出力するためには[0., 1]にしなければならない
# float型はガンマ補正という補正方法を用いるのだがこれが[0, 1]の範囲のためこのような作業が必要となっている(ここら辺はあまり考えずに受け入れて良い)


# 画素値をいじる
# 画素値はRGBで並んでいる．この場合(x=30、y=20の左からRGBが出力される)
# 最後の0はチャンネルが0である．つまりRを指定しし255としている(最も赤みを強くしている)．
# img[20:30, 100:150, 0] = 255
plt.figure()
ax = plt.subplot()
# ax.imshow(img)
# plt.show()


##################################################
# 以下チュートリアルの問題(左上1/4のRGBを反転させてBGRに)
##################################################
img3 = img_original.copy()
h, w, c = img3.shape
img3[: h//2, : w//2] = img3[: h//2, : w//2, ::-1]
ax.imshow(img3)
plt.show()