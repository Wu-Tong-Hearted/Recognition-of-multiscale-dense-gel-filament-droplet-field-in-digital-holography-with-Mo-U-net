import os

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import datetime

# ------------------------------------------------------------------
# labelme V2.0
# 功能：单像素修改功能，局部放大功能,剪裁图片,自动划分模式，涂画模式
# 使用方法
# 1. 配置路径，配置保存路径，修改图片
# z：修改显示模式，覆盖模式，边缘勾勒模式
# g：选择ROI区域，在changedesk窗口内进行单像素修改（会自动同步）
# a：add模式，左键单击增加前景像素
# e: era模式，左键单击减少前景像素
# f：flip模式，左键单击翻转像素的前后景
# s：保存修改后的label
# q: 退出程序
# c: 剪裁模式
# o：自动划分模式
#   ijkl分别控制对应框上下左右移动
# b: 调整画框变大（每次放大1）
# m：调整画框变小（每次缩小1，不可小于2倍）
# pen：调整画笔大小
# n：调整划分密度
# 
# 注：
# 1. 覆盖模式下可以调整gamma调整覆盖层的透明度
# 2. 按下左键拖动可以连续修改label
# 
# 剪裁图片功能
# 1. 按下c键，进入剪裁模式，设定cut width 表示保存下来的图片宽度大小（100-1200）
# 2. 用鼠标框选需要的部分，然后回车
# 3. 文件将被保存在cut文件下方
#
# 2021.6.28 wxyice
# ------------------------------------------------------------------

def change_pixel(mode,x,y,x0,y0,img_label,pen=1):

    if mode=='flip':
        if img_label[y//s+y0,x//s+x0]>127:
            value=0
        elif img_label[y//s+y0,x//s+x0]<127:
            value=255
        if pen==0:
            img_label[y//s+y0,x//s+x0]=value
        else:
            img_label[y//s+y0-pen:y//s+y0+pen,x//s+x0-pen:x//s+x0+pen]=value
        print('the x:y={0:3d}:{1:3d} change to {2:3d}'.format(x0+x//s,y0+y//s,value))
    elif mode=='add':
        value=255
        if pen==0:
            img_label[y//s+y0,x//s+x0]=value
        else:
            img_label[y//s+y0-pen:y//s+y0+pen,x//s+x0-pen:x//s+x0+pen]=value
        print('the x:y={0:3d}:{1:3d} change to {2:3d}'.format(x0+x//s,y0+y//s,255))
    elif mode=='era':
        value=0
        if pen==0:
            img_label[y//s+y0,x//s+x0]=value
        else:
            img_label[y//s+y0-pen:y//s+y0+pen,x//s+x0-pen:x//s+x0+pen]=value
        print('the x:y={0:3d}:{1:3d} change to {2:3d}'.format(x0+x//s,y0+y//s,0))


def onmouse_for_change(event, x, y, flags, param):  # 创建回调函数
    global x0,y0,s,ix, iy,save
    global mode,pen

    # 单像素修改
    if event == cv2.EVENT_LBUTTONDOWN:  # 按下左键
        #print(x,y,'----------')
        change_pixel(mode,x,y,x0,y0,img_label,pen)
        ix,iy=x,y
        save=False
    elif event == cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        temp=mode
        if mode=='flip':
            mode='add'
        change_pixel(mode,x,y,x0,y0,img_label,pen)
        mode=temp
def print_gamma(x):
    pass
 

def format_time(time):
    return '{0:04d}{1:02d}{2:02d}_{3:02d}_{4:02d}_{5:2d}'.format(time.year,time.month,time.day,time.hour,time.minute,time.second)

def autocut(n,H,W):
    dH,dW=H//n,W//n
    Hp=[dH*i for i in range(n)]
    Wp=[dW*i for i in range(n)]
    rect=[]
    for h in Hp:
        for w in Wp:#xywh box
            rect.append([w,h,dW,dH])
    return rect


if __name__ == '__main__':
    # 选择一张图像，配置input和true的路径路径,配置修改后label的保存路径
    path_of_img=r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_2.58mm velocity_118.2\cut_672\test_img\img5.jpg'
    path_of_label=r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_2.58mm velocity_118.2\cut_672\ground_truth\gt5.jpg'
    path_of_pred=r'src_photo\enhance_data_jpg\train_label\00001.jpg'

    save_label_path=r'C:\Users\Pangzhentao\learn_keras\test data\final_test\true_label_by_hand\diameter_2.58mm velocity_118.2\cut_672\ground_truth\gt5.jpg'
    save_cut_path=r'cut'

    if os.path.exists(save_cut_path)!=True:
        os.mkdir(save_cut_path)

    name_of_window='labelme'
    name_of_change='changedesk'
    name_of_scrope='scrope'

    # 读取图像
    img_src=cv2.imread(path_of_img)/255 # 0-1 方便叠加 3channal
    img_label=cv2.imread(path_of_label,0) # 1channal
    if path_of_pred!=None:
        img_pred=cv2.imread(path_of_pred,0)

    gamma=0.2 # 叠加阈值
    mode_of_edge=False

    cv2.namedWindow(name_of_window)                 # 创建空窗口
    cv2.namedWindow(name_of_change)

    cv2.setMouseCallback(name_of_change, onmouse_for_change)  # 将回调函数与窗口绑定

    cv2.createTrackbar('gamma',name_of_window,0,100,print_gamma) 
    cv2.createTrackbar('cut_width',name_of_window,10,1200,print_gamma) 
    cv2.createTrackbar('pen',name_of_change,1,5,print_gamma) 

    cv2.createTrackbar('n',name_of_change,4,8,print_gamma) 

    # 初始化
    ix,iy=-1,-1
    x0,y0=0,0
    s=1
    ROI=None
    initBB=None
    pen=1
    mode='flip'  # 默认翻转模式
    save=False
    size_big=5
    padding=20
    auto=False
    numbox=1
    n=4
    temp=n
    H,W=img_src.shape[0],img_src.shape[1]
    s=2
    
    rect=autocut(n,H,W)

    while True:
        if mode_of_edge:
            # 边缘显示模式
            img=img_src.copy() # 保留原图 
            img=img*255
            img = img.astype(np.uint8)
            edge_of_label=cv2.Canny(img_label,255,255)
            img[edge_of_label>127]=[255,255,255] # 标记边缘为白色 
        else:
            #表示覆盖模式
            img=img_src.copy()
            img_mask=img_src.copy()
            img_mask[img_label>127]=[0,0,1]
            img=(img*gamma+img_mask*(1-gamma))*255
            img = img.astype(np.uint8)

        key=cv2.waitKey(1) & 0xFF   #获取键值
        gamma = cv2.getTrackbarPos('gamma',name_of_window)
        cut_width=cv2.getTrackbarPos('cut_width',name_of_window)
 
        n=cv2.getTrackbarPos('n',name_of_change)
        if temp!=n:
            if n==0 or n==1:
                n=2
            rect=autocut(n,H,W)
            temp=n
        pen = cv2.getTrackbarPos('pen',name_of_change)

        gamma = gamma/100

        if key==ord('z'):# 按下z键，交换边缘显示模式和全局显示模式
            mode_of_edge= not mode_of_edge
        elif key==ord('q'): # 按下q键，退出程序
            break
        elif key==ord('g'): # 按下g键，进入选择小框
            initBB = cv2.selectROI(name_of_window,img, fromCenter=False,showCrosshair=False) 
        elif key==ord('a'): # 按下a键，进入add模式
            mode='add'
        elif key==ord('e'): # 按下e键，进入擦除模式 
            mode='era'
        elif key==ord('f'): # 按下f键进入翻转模式
            mode='flip'
        elif key==ord('s'):
            cv2.imwrite(save_label_path,img_label)
            save=True
            
        elif key==ord('o'):
            auto= not auto
        elif key==ord('u'):  # 按下u键 刷新
            img_src=cv2.imread(path_of_img)/255 
            cv2.imwrite(save_label_path,img_label)
            img_label=cv2.imread(save_label_path,0) 
        elif key==ord('m'):
            s-=1
            if s<2:
                s=2
        elif key==ord('b'):
            s+=1
        elif key==ord('c'): # 进入cut 模式
            mode='cut'

            img=img_src.copy()
            img=img*255
            img = img.astype(np.uint8)
            img[img_label>127]=[255,255,255]
            img[img_pred>127]=[0,0,180]
            cv2.imshow(name_of_window,img)
            cutROI=cv2.selectROI(name_of_window,img,fromCenter=False,showCrosshair=True)
            x_cut,y_cut,w_cut,h_cut=cutROI
            
            # cut
            img=img_src.copy()
            img=img*255
            img = img.astype(np.uint8)
            inputimg=img[y_cut:y_cut+h_cut,x_cut:x_cut+w_cut]
            predimg=img_pred[y_cut:y_cut+h_cut,x_cut:x_cut+w_cut]
            adthimg=img_label[y_cut:y_cut+h_cut,x_cut:x_cut+w_cut]

            # resize
            s_cut=max(1,cut_width//w_cut)

            inputimg=cv2.resize(inputimg,(round(inputimg.shape[1]*s_cut),round(inputimg.shape[0]*s_cut)))
            predimg=cv2.resize(predimg,(round(predimg.shape[1]*s_cut),round(predimg.shape[0]*s_cut)))
            adthimg=cv2.resize(adthimg,(round(adthimg.shape[1]*s_cut),round(adthimg.shape[0]*s_cut)))

            # get_time
            time=datetime.datetime.now()
            time=format_time(time)
            

            # save
            cv2.imwrite(os.path.join(save_cut_path,'{0}_inputcut.jpg'.format(time)),inputimg)
            cv2.imwrite(os.path.join(save_cut_path,'{0}_predcut.jpg'.format(time)),predimg)
            cv2.imwrite(os.path.join(save_cut_path,'{0}_adthcut.jpg'.format(time)),adthimg)
            # 恢复flip模式
            mode='flip'

        if auto: # 自动划分
            if key==ord('l'):
                numbox+=1
                numbox=min(numbox,n**2)
            elif key==ord('j'):
                numbox-=1
                numbox=max(numbox,1)
            elif key==ord('k'):
                numbox+=n
                numbox=min(numbox,n**2)
            elif key==ord('i'):
                numbox-=n
                numbox=max(numbox,1)
            try:
                x0,y0,w0,h0=rect[numbox-1]
            except :
                rect=autocut(n,H,W)

            cv2.rectangle(img,(x0,y0),(x0+w0,y0+h0),[255,255,255],thickness=1)
            ROI=[x0,y0,w0,h0]
        else:
            if initBB!=None:
                x0,y0,w0,h0=initBB
                print(initBB)
                ROI=img[y0:y0+h0,x0:x0+w0]
                s=500//ROI.shape[0]
                ROI=cv2.resize(ROI,(ROI.shape[1]*s,ROI.shape[0]*s))
                initBB=None
        if ROI is not None:
            ROI=img[y0:y0+h0,x0:x0+w0]
            #s=500//ROI.shape[0]
            ROI=cv2.resize(ROI,(ROI.shape[1]*s,ROI.shape[0]*s))
            cv2.imshow(name_of_change,ROI)
        # 输出信息
        info = [
            ('control','add=a,flip=f,era=e,quit=q,save=s,mode_of_edge=z,get ROI=g'),
            ("image name",path_of_img.split(r"/")[-1]),
            ("save",save),
            ("mode",mode),
            ("shape", img_src.shape[:2]),
            ("mode_of_edge",mode_of_edge)
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img, text, (10, ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(name_of_window,img)

    cv2.destroyAllWindows()



    

    
    
    
  

