import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



#Using Depth images to train model now
#def bgr2rgb(image):
#    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def crop(image):
    cropped = image[ 300:400, :]
    return cropped	

def flip(image):
    return cv2.flip(image,1)

def resize(image, shape=(160, 70)):
    return cv2.resize(image, shape)


def checker(adr2, adr1):
    adr = '_out/' + adr2 + '/CameraDepth/' + str(adr1)
    img = mpimg.imread(adr)   
    
    return img

def flipper(adr1,adr2):
    adr = '_out/' + adr1 + '/CameraDepth/' + str(adr2)
    img = mpimg.imread(adr)
    flipped = flip(img)	
    
    return flipped


if __name__ == '__main__':
    path0 = 'Data/'

    for i in range(900):
        str4 = '{:06d}'.format(i) + '.png'
        count  = 0
        for j in range(3):
            img_check = checker('episode_000' + str(j),str4)
            img_flip = flipper('episode_000' + str(j),str4)
            # cv2.imwrite(os.path.join(path0 ,'check' + str(i) + '.png'), img_check)
            print(path0 + 'check' + str((900*count)+i) + '.png')
            plt.imsave(path0 + 'check' + str((900*count)+i) + '.png',img_check)
            plt.imsave(path0 + 'flip' + str((900*count)+i) + '.png',img_flip)
            count+=1

