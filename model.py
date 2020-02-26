import csv
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.layers import Convolution2D,Lambda,Cropping2D
from keras.layers import Activation,BatchNormalization
import argparse
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split

def network():
    model = Sequential()

    # Normalization Layer
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=(70, 160, 3)))
      
#     # trim image to only see section with road
#     model.add(Cropping2D(cropping=((70,25),(0,0)))) 

    # Convolutional Layer 1
    model.add(Convolution2D(filters=24, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 2
    model.add(Convolution2D(filters=36, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 3
    model.add(Convolution2D(filters=48, kernel_size=5, strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 4
    model.add(Convolution2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Convolutional Layer 5
    model.add(Convolution2D(filters=64, kernel_size=3, strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Flatten Layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 2
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully Connected Layer 3
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam')
    return model

class TopLevel:
    def __init__(self, model, base_path='', epoch_count=2):
        self.data = []
        self.model = model
        self.epochs = epoch_count  #tune this
        self.base_path = base_path
        self.correction_factor = 0.2
        self.image_path = self.base_path + '/IMG/'
        self.driving_log_path = self.base_path + '/driving_log.csv'
        self.training_samples = []
        self.validation_samples = []
        self.batch_size = 128  #tune this    

    #split train data into training and validation sets

    def split_data(self,split_ratio):
        self.training_samples, self.validation_samples = train_test_split(self.data,test_size= split_ratio)

        return None

    #read data from simulator

    def read_data(self):
        with open(self.driving_log_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                self.data.append(line)
#         print(self.data)
        return None

    #data augmentation tactics

    def bgr2rgb(self,image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    def crop(self,image):
        cropped = image[ 60:130, :]
        return cv2.resize(cropped,(160,70))

    def flip(self,image):
        return cv2.flip(image,1)
    
    def resize(self,image, shape=(160, 70)):
        return cv2.resize(image, shape)
    
    def crop_resize(self,image):
        cropp = self.crop(image)
        resized = self.resize(cropp)
        return resized
    
#     def process_image(self,batch_sample):
    #         steering_angle = np.float32(batch_sample[)
#         images, steering_angles = [][]

#         for image_path_index in rae(3):
#             image_name = batch_sample[image_path_index].split(')[-1]

#             image = cv2.imread(self.image_path + age_name)
#             rgb_image = self.b2rgb(image)
#             resized = self.crop_rese(rgb_image)

#             imagesppend(resized)

#             if ima_path_index == 1:
#                 steering_angles.append(steering_angle + se.correction_factor)
#             elifmage_path_index == 2:
#                 steering_angles.append(steering_angle self.correction_ctor)
#             else:
#                 steering_ales.append(steering_angle)

#           if image_path_index == 0:
#                 flipped_nter_image = self.flip(resized)
#                 ages.append(flipped_center_image)
#                 steering_angles.appd(-steering_angle)
                
#         return images, steering_angles
        

    #top level for augmentation

    def generator(self, samples, batch_size=128):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(1, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    for i in range(0,3): #we are taking 3 images, first one is center, second is left and third is right
                        
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) #since CV2 reads an image in BGR we need to convert it to RGB since in drive.py it is RGB
                        center_angle = float(batch_sample[3]) #getting the steering angle measurement
                        images.append(self.crop_resize(center_image))
                        

                        
                        # introducing correction for left and right images
                        # if image is in left we increase the steering angle by 0.2
                        # if image is in right we decrease the steering angle by 0.2
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        # Code for Augmentation of data.
                        # We take the image and just flip it and negate the measurement
                        
                        images.append(self.crop_resize(cv2.flip(center_image,1)))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
#here we got 6 images from one image  
                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

    def train_generator(self):
        return self.generator(samples=self.training_samples, batch_size=128)

    def validation_generator(self):
        return self.generator(samples=self.validation_samples, batch_size=128)

    def run(self):
        self.model.fit_generator(generator=self.train_generator(), 
                                 samples_per_epoch= len(self.training_samples),
                                 validation_data=self.validation_generator(), 
                                 nb_val_samples=len(self.validation_samples),
                                 epochs=self.epochs,
                                 verbose=1)
        self.model.save('model.h5')

def main():
    parser = argparse.ArgumentParser(description='Train a car to drive itself')
    parser.add_argument(
        '--data-base-path',
        type=str,
        default='./data',
        help='Path to image directory and driving log')

    args = parser.parse_args()

    # Instantiate the pipeline
    pipeline = TopLevel(model=network(), base_path=args.data_base_path, epoch_count=3)
    pipeline.read_data()
    pipeline.split_data(split_ratio=0.2)
#     print(len(pipeline.training_samples))
#     print(len(pipeline.validation_samples))
#     print(len(pipeline.training_samples))
#     print(len(pipeline.validation_samples))
    pipeline.run()
    print(pipeline.model.get_weights()) 
    
if __name__ == '__main__':   
    main()
