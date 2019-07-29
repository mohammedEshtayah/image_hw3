import cv2
import numpy as np
import sys
import math

def Multiply_A(Original_Image):
    org_image=cv2.imread(Original_Image,0).astype(np.uint8)
    new_filter=np.zeros((height,width))
    height = np.size(org_image, 0)
    width = np.size(org_image, 1)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            new_image[i][j]=(-1)^i+j*org_image[i][j]

    #cv2.imshow('Original image', new_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    org_image=cv2.imread(Original_Image,0).astype(np.uint8)
    height = np.size(org_image, 0)
    width = np.size(org_image, 1)
    new_image=np.zeros((height,width))
    avg=height*width
    for u in range( height):
        for v in range( width) :
            sum=0
            for x in range( height ):
                for y in range( width ):
                   sum+=org_image[x][y] * math.exp(- 2 *np.pi*( u*x /height + v*y  / width))
 
            new_image[u][v]=sum/avg
def Lowpass(Original_Image):
    D0=30
    org_image=cv2.imread(Original_Image,0).astype(np.uint8)
    f = np.fft.fft2(org_image)
    
    fshift = np.fft.fftshift(f)
    height = np.size(org_image, 0)
    width = np.size(org_image, 1)
    m2,n2 = height/2 , width/2
    d=np.zeros((height,width))
    h=np.zeros((height,width))

    for u in range( height ):
        for v in range( width ):
            r=((u-m2)**2)+((v-n2)**2)
            d[u][v]=math.sqrt(r)
            d[u][v]=int(d[u][v])
            if d[u][v]>D0:
                h[u][v]=0
            elif d[u][v]<=D0:
                h[u][v]= math.exp(-(d[u][v]**2)/(2*(D0**2) ) ) 

           #new_filter[x][y]= int(math.exp( - ( (org_image[x][y]**2) / ((2* D0**2 )) ) ))
   
    result= h*fshift  
    f_ishift = np.fft.ifftshift(result)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    cv2.imshow('Original image',img_back.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Highpass(Original_Image):
    D0=200
    org_image=cv2.imread(Original_Image,0).astype(np.uint8)
    f = np.fft.fft2(org_image)
    fshift = np.fft.fftshift(f)
    height = np.size(org_image, 0)
    width = np.size(org_image, 1)
    m2,n2 = height/2 , width/2
    d=np.zeros((height,width))
    h=np.zeros((height,width))

    for u in range( height ):
        for v in range( width ):
            r=((u-m2)**2)+((v-n2)**2)
            d[u][v]=math.sqrt(r)
            d[u][v]=int(d[u][v])
            if d[u][v]>D0:
                h[u][v]=0
            elif d[u][v]<=D0:
                h[u][v]=1- math.exp(-(d[u][v]**2)/(2*(D0*D0) ) ) 

          
    result= h*fshift  
    f_ishift = np.fft.ifftshift(result)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    cv2.imshow('Original image',img_back.astype(np.uint8) )
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
def Spectrum_Average(Original_Image):
    org_image=cv2.imread(Original_Image,0).astype(np.uint8)
    f = np.fft.fft2(org_image)
    fshift = np.fft.fftshift(f)
    cv2.imshow('Original image',fshift.astype(np.uint8) )
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
    height = np.size(org_image, 0)
    width = np.size(org_image, 1) 
    avg=height*width
    d=np.zeros(( np.size(fshift, 0), np.size(fshift, 1) ))
    for u in range( height ):
        for v in range( width ):
            fshift[u][v]=fshift[u][v]

def Inverse_Fourier_transform(Original_Image):
    org_image=cv2.imread(Original_Image,0).astype(np.uint8)
    height = np.size(org_image, 0);  width = np.size(org_image, 1)
    new_image=np.zeros((height,width))
    for u in range( height):
        for v in range( width) :
            sum=0
            for x in range( height ):
                for y in range( width ):
                    sum+=org_image[x][y] * np.exp(2 *np.pi* ( u*x /height +   v*y / width))
 
            new_image[u][v]=int(sum) 

def Fourier_transform(Original_Image):
    org_image=cv2.imread(Original_Image,0).astype(np.uint8)
    height = np.size(org_image, 0);  width = np.size(org_image, 1)
    new_image=np.zeros((height,width))
    for u in range( height):
        for v in range( width) :
            sum=0
            for x in range( height ):
                for y in range( width ):
                    sum+=org_image[x][y] * np.exp(-2j *np.pi* ( u*x /height +   v*y / width))
 
            new_image[u][v]=int(sum) 

if __name__ == "__main__":

    Original_Image = str(sys.argv[1])
 
    #new_image = Multiply_A(Original_Image)
    #f = np.fft.ifft2(org_image)
    #Fourier_transform(Original_Image)
    #Inverse_Fourier_transform(Original_Image)
    #Lowpass(Original_Image)
    Spectrum_Average(Original_Image) 
    #Highpass(Original_Image)
    #cv2.imshow('After applying the filter', new_image)
    #Original_Image = cv2.imread(Original_Image).astype(np.uint8)
    #cv2.imshow('Original image', Original_Image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
