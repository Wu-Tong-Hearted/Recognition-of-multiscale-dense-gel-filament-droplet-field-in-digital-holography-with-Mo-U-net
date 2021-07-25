import os

import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import datetime

# ------------------------------------------------------------------
# labelme V2.0
# Functions: 
# single pixel modification function
# local magnification function
# clipping picture
# automatic division mode
# painting mode
# 
# Instructions
# 1. Configure path, configure save path, modify picture
# 
# z：modify display mode, overlay mode and edge outline mode
# g：select the ROI area (synchronized automatically)
# a：add Mode, left click to increase foreground pixels
# e: era Mode, left click to reduce foreground pixels
# f：flip Mode, left click to flip the foreground and background of pixels
# s：Save the modified label
# q: quit
# c: cut
# o：automatic  mode
#   ijkl  controls the corresponding box to move up, down, left and right respectively
# b: adjust the frame to enlarge (1 at a time)
# m：adjust the picture frame to become smaller (reduce it by 1 at a time, not less than 2 times)
# pen：adjust pen size
# n：adjust partition density
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


def onmouse_for_change(event, x, y, flags, param):  
    global x0,y0,s,ix, iy,save
    global mode,pen


    if event == cv2.EVENT_LBUTTONDOWN:  
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

    
    img_src=cv2.imread(path_of_img)/255
    img_label=cv2.imread(path_of_label,0) 
    if path_of_pred!=None:
        img_pred=cv2.imread(path_of_pred,0)

    gamma=0.2 
    mode_of_edge=False

    cv2.namedWindow(name_of_window)                 
    cv2.namedWindow(name_of_change)

    cv2.setMouseCallback(name_of_change, onmouse_for_change)  

    cv2.createTrackbar('gamma',name_of_window,0,100,print_gamma) 
    cv2.createTrackbar('cut_width',name_of_window,10,1200,print_gamma) 
    cv2.createTrackbar('pen',name_of_change,1,5,print_gamma) 

    cv2.createTrackbar('n',name_of_change,4,8,print_gamma) 

    # init
    ix,iy=-1,-1
    x0,y0=0,0
    s=1
    ROI=None
    initBB=None
    pen=1
    mode='flip'  
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
            # Edge display mode
            img=img_src.copy() 
            img=img*255
            img = img.astype(np.uint8)
            edge_of_label=cv2.Canny(img_label,255,255)
            img[edge_of_label>127]=[255,255,255] 
        else:
            # overlay mode
            img=img_src.copy()
            img_mask=img_src.copy()
            img_mask[img_label>127]=[0,0,1]
            img=(img*gamma+img_mask*(1-gamma))*255
            img = img.astype(np.uint8)

        key=cv2.waitKey(1) & 0xFF   
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


        # Press the Z key to switch between the edge display mode and the global display mode
        if key==ord('z'):
            mode_of_edge= not mode_of_edge
        elif key==ord('q'): # Press the Q key to exit the program
            break
        elif key==ord('g'): # Press the G key to enter the selection box
            initBB = cv2.selectROI(name_of_window,img, fromCenter=False,showCrosshair=False) 
        elif key==ord('a'): # Press the a key to enter the add mode
            mode='add'
        elif key==ord('e'): # Press the e key to enter the erase mode
            mode='era'
        elif key==ord('f'): # Press the F key to enter the flip mode
            mode='flip'
        elif key==ord('s'):
            cv2.imwrite(save_label_path,img_label)
            save=True
            
        elif key==ord('o'):
            auto= not auto
        elif key==ord('u'):  # Press the U key to refresh
            img_src=cv2.imread(path_of_img)/255 
            cv2.imwrite(save_label_path,img_label)
            img_label=cv2.imread(save_label_path,0) 
        elif key==ord('m'):
            s-=1
            if s<2:
                s=2
        elif key==ord('b'):
            s+=1
        elif key==ord('c'): # Enter cut mode
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
            mode='flip'

        if auto: 
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
        # Output information
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