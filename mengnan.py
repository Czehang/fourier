'''
import cv2

cap=cv2.VideoCapture("D:/Image/mennan.mp4")
c=0

while(cap.isOpened()):
    c=c+1
    value,image=cap.read()
    if value:
        cv2.imwrite('D:/Image/mengnan'+str(c)+'.jpg',image)
        cv2.waitKey(1)
    else:
        break
cap.release()'''
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import filters,zoom
from scipy import signal
from scipy import optimize
c=0
plt.figure(figsize=(13,5),dpi=80)
###############函数显示
ROBERTS_A=np.mat("1 0;0 -1")
ROBERTS_B=np.mat("0 1;-1 0")
def f(x):
    return 2*np.sin(x)+3
def f_fit(x,a,b):
    return a*np.sin(x)+b
def f_show(x,p_fit):
    a,b=p_fit.tolist()
    return a*np.sin(x)+b

def fourier(x, *a):
    e=np.zeros(len(x))+a[0]+a[len(a)//2];
    for n in range(1,len(a)//2):
        e += a[n] * np.cos(n*x)+a[len(a)//2+n]*np.sin(n*x)
    return e

def around(clayer,tolayer):
    for layer in range(clayer,tolayer+1):
        for R in range(-layer,layer+1):
            for L in range(-layer,layer+1):
                if abs(R)==layer or abs(L)==layer:
                    yield R,L
def lins(x,y,linx,liny):
    mapxy=set((x[n],y[n])for n in range(min(len(x),len(y))))
    def find(x,y):
        xx,yy=[],[]
        while len(mapxy):
            for R,L in around(1,5):
                px,py=x+R,y+L
                if px<=linx and py<=liny and (px,py) in mapxy:
                    xx.append(px)
                    yy.append(py)
                    mapxy.remove((px,py))
                    x,y=px,py
                    break
            else:
                return xx,yy
        return xx,yy
    while len(mapxy):
        x,y=next(iter(mapxy))
        mapxy.remove((x,y))
        linsx,linsy=find(x,y)
        linsx2,linsy2=find(x,y)
        linsx.reverse()
        linsy.reverse()
        lx,ly=linsx+[x]+linsx2,linsy+[y]+linsy2
        print('len=',len(mapxy),len(lx),' '*2,end='\r')
        yield lx,ly
def edge_demo(image):
    image = image.mean(axis = -1) #RGB值平均，转成灰度图
    image=filters.gaussian_filter(image,2) #高斯降噪
    image=zoom(image,.5)
    image=(image>128)+0 #阈值
    
    # plt.imshow((image>128)+0,cmap='gray')
    edge_output=np.hypot(signal.convolve2d(image,ROBERTS_A,mode='same'),signal.convolve2d(image,ROBERTS_B,mode='same')) # Roberts卷积
    plt.subplot(1,2,2)
    plt.imshow(edge_output,cmap='gray')
    mapy,mapx=np.where(edge_output!=0)
    for x,y in lins(mapx,mapy,edge_output.shape[1],edge_output.shape[0]):
        if len(x)>10:
            x,y=np.array(x),np.array(y)
            T=np.linspace(0,2*np.pi,len(x))
            poptx,pcovx=optimize.curve_fit(fourier,T,x,[1.0] * (len(x)//10))
            popty,pcovy=optimize.curve_fit(fourier,T,y,[1.0] * (len(y)//10))
            plt.plot(fourier(T, *poptx),fourier(T, *popty))
    plt.pause(0.01)
    plt.clf()
while True:
    c=c+1
    image=cv.imread('D:/image/mengnan'+str(c)+'.jpg')
    plt.subplot(1,2,1)
    plt.imshow(image,'gray')
    edge_demo(image)
    # src = cv.imread('D:/image/mengnan10.jpg')
    # src = cv2.imread('D:/image/mengnan'+str(c)+'.jpg')
    # if (image.shape==None):
    #     break
    # f=np.fft.fft2(image)
    # fshift=np.fft.fftshift(f)
    # fimg=np.log(np.abs(fshift))
    # #设置高通滤波器
    # rows, cols = image.shape
    # crow,ccol = int(rows/2), int(cols/2)
    # fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
    # #傅里叶逆变换
    # ishift = np.fft.ifftshift(fshift)
    # iimg = np.fft.ifft2(ishift)
    # iimg = np.abs(iimg)

# ###########Canny边缘检测C
#     blurred = cv2.GaussianBlur(src, (3, 3), 0)
#     gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
#     edge_output = cv2.Canny(gray, 50, 150)






plt.ioff()



#     plt.subplot(1,2,1)
#     plt.imshow(image,'gray')
#     plt.subplot(1,2,2)
#     plt.imshow(edge_output,'gray')
#     #plt.imshow(image,'seismic')
#     plt.pause(0.01)
#     plt.clf()
# plt.ioff()

