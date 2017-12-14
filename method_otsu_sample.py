import cv2
import numpy as np
from mplotlib import pyplot as plt
	
#загружаем и отображаем избражение
image = cv2.imread('image.jpg')   
cv2.imshow('Hi',image)
cv2.waitKey(0)

image.shape
red = image[:,:,2] #только красная компонента
cv2.imshow("red image",red) #отображение графика
cv2.waitKey(0)

#пример использования  функций OpenCV. Метод Оцу
# глобальное пороговое значение
ret1, th1 = cv2.threshold (channel2, 127,255, cv2.THRESH_BINARY)
# Потенциал Otsu
ret2, th2 = cv2.threshold (channel2, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Порог Otsu после гауссовой фильтрации
blur = cv2.GaussianBlur (channel2, (5,5), 0)
ret3, th3 = cv2.threshold (blur, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
# отобразите все изображения и их гистограммы
images = [channel2, 0, th1,
          channel2, 0, th2,
          blur, 0, th3]
title = [ 'Original Noisy Image' , 'Histogram' , 'Global Thresholding (v = 127)' ,
           'Оригинальное шумовое изображение' , 'Гистограмма' , 'Порог Отсу' ,
           'Гауссово фильтрованное изображение' , 'Гистограмма' , 'Порог Отсу' )

   for i in xrange(3):
       plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
       plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
       plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
       plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
       plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
       plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
 plt.show()
 
 
 #Реализация метода Оцу
#hist, bins = np.histogram(red, bins=255) # гистограмма до 
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist)
#plt.show()
a = red.ravel()
k = 255./np.max(red)
red_new = []
for row in red:
    row1 = []
    for el in row:
        row1.append(int(el*k))
    red_new.append(row1) 
red_new = np.array(red_new,dtype=np.uint8)     
#cv2.imshow("red image",red) #отображение графика
#cv2.waitKey(0)
#plt.figure()
hist, bins = np.histogram(red_new, bins=255) # гистограмма
#center = (bins[:-1] + bins[1:]) / 2
#plt.bar(center, hist)
#plt.show()
#cv2.imwrite('final.png',np.array(red_new)) # линейное контрастирование
print len(bins), len(hist)
sigma_max = 0
T = 0
sum_all = 0
for i in range(len(hist)):
    sum_all+=hist[i]*bins[i]
"""for t in range(1,255-2):
    w1 = sum(hist[0:t])/sum_all
    w2 = sum(hist[t+1:-1])/sum_all
    a1 = sum(hist[0:t]*bins[0:t])/sum(hist[0:t])
    #print t,hist[t+1:-1]
    a2 = sum(hist[t+1:-1]*bins[t+1:-2])/sum(hist[t+1:-1])
    #print w1, w2, a1,a2
    sigma = (w1*w1*(a1-a2)**2)
    #print sigma
    if sigma > sigma_max:
        sigma_max = sigma
        T = t"""
wb = 0  
sumB =0   
varMax =0  
total = len(red.ravel()) 
for i in range(len(hist)):
    wb += hist[i]/sum_all  
    print wb
    if wb == 0:
        continue
    wf = total - hist[i]
    if wf == 0:
        break
    sumB += float(i*hist[i])
    mB = sumB/wb
    mF = (sum_all-sumB) / wf
    varBetween = wb*wf*(mB-mF)**2
    if varBetween > varMax:
        varMax = varBetween
        T = i
    
print T
bw1 = T
red_new_th = []      
for row in red:
    row1 = []
    for el in row:
        if el >= bw1: 
            row1.append(255)
            #print el, T, "255"
        else:
            row1.append(0)
            #print el, T, "0"
    red_new_th.append(row1) 
red_new_th = np.array(red_new_th)
cv2.imwrite('my_otsu.png',red_new_th) # метод Оцу

#ret1,th1 = cv2.threshold(red,0,255,cv2.THRESH_OTSU)
#cv2.imwrite('not_my_otsu_1.png',th1)
#ret2,th2 = cv2.threshold(red_new,0,255,cv2.THRESH_OTSU)
#cv2.imwrite('not_my_otsu_2.png',th2)
#xor = cv2.bitwise_xor(th1,th2)
