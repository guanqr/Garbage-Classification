# 方法一：knn检测方法
# import cv2
# import numpy as np
# knn = cv2.createBackgroundSubtractorKNN(detectShadows = True)
# es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,12))
# # camera = cv2.VideoCapture("movie.mpg")
# camera = cv2.VideoCapture(0)
# def drawCnt(fn, cnt):
#   if cv2.contourArea(cnt) > 1400:
#     (x, y, w, h) = cv2.boundingRect(cnt)
#     cv2.rectangle(fn, (x, y), (x + w, y + h), (255, 255, 0), 2)
 
# while True:
#   ret, frame = camera.read()
#   cv2.imshow("Capture_Test",frame)

#   if not ret:
#     continue
#   fg = knn.apply(frame.copy())
#   fg_bgr = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
#   bw_and = cv2.bitwise_and(fg_bgr, frame)
#   draw = cv2.cvtColor(bw_and, cv2.COLOR_BGR2GRAY)
#   draw = cv2.GaussianBlur(draw, (21, 21), 0)
#   draw = cv2.threshold(draw, 10, 255, cv2.THRESH_BINARY)[1]
#   draw = cv2.dilate(draw, es, iterations = 2)
#   print(draw)
#   image, contours, hierarchy = cv2.findContours(draw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   for c in contours:
#     drawCnt(frame, c)
#   cv2.imshow("motion detection", frame)
#   if cv2.waitKey(int(1000 / 12)) & 0xff == ord("q"):
#     break

# camera.release()
# cv2.destroyAllWindows()

#方法二：mog2方法
# import cv2
# import numpy as np

# camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头
# bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
# es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# while True:
#     grabbed, frame_lwpCV = camera.read()
#     if not grabbed:
#       continue
#     fgmask = bs.apply(frame_lwpCV) # 背景分割器，该函数计算了前景掩码
#     # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值，在阈值化图像中，将非纯白色（244~255）的所有像素都设为0，而不是255
#     th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
#     # 下面就跟基本运动检测中方法相同，识别目标，检测轮廓，在原始帧上绘制检测结果
#     dilated = cv2.dilate(th, es, iterations=2) # 形态学膨胀
#     image, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
#     for c in contours:
#         if cv2.contourArea(c) > 1600:
#             (x, y, w, h) = cv2.boundingRect(c)
#             cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 255, 0), 2)

#     cv2.imshow('mog', fgmask)
#     cv2.imshow('thresh', th)
#     cv2.imshow('detection', frame_lwpCV)
#     key = cv2.waitKey(1) & 0xFF
#     # 按'q'健退出循环
#     if key == ord('q'):
#         break
# # When everything done, release the capture
# camera.release()
# cv2.destroyAllWindows()


#方法三：grabcut方法
import numpy as np
import cv2
import time  # 引入time模块
cap = cv2.VideoCapture(0)
# ticks = time.time()
# # 读入图片
# img = cv2.imread("./car.jpg")
while True:
  ret,frame = cap.read()
  if not ret:
      continue
  cv2.imshow("Capture_Test",frame)
# 创建一个和加载图像一样形状的 填充为0的掩膜
  img=frame
  mask = np.zeros(img.shape[:2], np.uint8)

  # 创建以0填充的前景和背景模型
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)

  # 定义一个矩形
  rect = (100, 50, 421, 378)

  cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
  mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
  img = img*mask2[:, :, np.newaxis]

  # print(time.time()-ticks)
  cv2.imshow("cut", img)
  # cv2.imshow('origin', cv2.imread("./car.jpg"))
  cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()

