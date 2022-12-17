import cv2
import numpy as np
#from scipy.misc import imresize, imread
from scipy import ndimage
from PIL import Image
from cv2 import imread

def Preprocess(img_path, filename, img_size=(64, 128)):
    img = imread(img_path+filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    cropped = crop_image(img)
    mask = remove_background(cropped)
    inverted = 255 - mask
    resized = resize_image(inverted, img_size)
    threshold, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('Mask', mask)
    #cv2.imshow('Resized', cv2.bitwise_not(resized))
    #cv2.imshow('Bin', cv2.bitwise_not(binary))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    return (binary/255).astype(int) #, resized   

def crop_image(img):
    blur_radius = 2
    blurred_image = ndimage.gaussian_filter(img, blur_radius)
    mask = remove_background(blurred_image)
    inverted = 255 - mask
    cnts = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = np.concatenate(cnts)
    x,y,w,h = cv2.boundingRect(cnts)
    cropped = img[y:y+h, x:x+w]
    return cropped

def remove_background(img):
    img = img.astype(np.uint8)
    threshold, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img[img > threshold] = 255
    return img

def resize_image(image, new_size, interpolation='bilinear'):
    height, width = new_size
    image = np.array(Image.fromarray(obj=image).resize(size=(width, height)))
    return image

if __name__ == '__main__':
    signature = Preprocess("./", "myimg.jpg")
    print(signature)
