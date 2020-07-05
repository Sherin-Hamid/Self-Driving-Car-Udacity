import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, Dense, Dropout, Flatten, Lambda, MaxPooling2D

lines = []
i=0
with open('./u_data/data/driving_log.csv') as f:
     reader = csv.reader(f)
     for line in reader:
        if(i != 0):
            lines.append(line)
        i += 1
  
images = []
angles = []
augmented_images = []
augmented_angles = []
          
for line in lines:
    # read in images from center, left and right cameras
    center_image = line[0]
    filename = center_image.split('/')[-1]
    current_path = './u_data/data/IMG/' + filename
    img_center = cv2.imread(current_path)
    
    left_image = line[1]
    filename = left_image.split('/')[-1]
    current_path = './u_data/data/IMG/' + filename
    img_left = cv2.imread(current_path)
    
    right_image = line[2]
    filename = right_image.split('/')[-1]
    current_path = './u_data/data/IMG/' + filename
    img_right = cv2.imread(current_path)
  
    #images.append(image)

    steering_center = float(line[3])
    #angles.append(angle)
   
    # create adjusted steering measurements for the side camera images
    correction_left = 0.4
    correction_right = 0.5
    steering_left = steering_center + correction_left
    steering_right = steering_center - correction_right
 
    # add images and angles to data lists
    images.extend([img_center, img_left, img_right])
    angles.extend([steering_center, steering_left, steering_right])
    

for image,angle in zip(images, angles):
    augmented_images.append(image)
    augmented_angles.append(angle)
    augmented_images.append(np.fliplr(image))
    augmented_angles.append(-angle)
    
    
#X_train = np.array(images)
#y_train = np.array(angles)
X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)

X_train, y_train = shuffle(X_train, y_train)
          
# Build the model
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24, kernel_size=5, subsample=(2, 2), activation = 'relu'))
model.add(Conv2D(36, kernel_size=5, subsample=(2, 2), activation = 'relu'))
model.add(Conv2D(48, kernel_size=5, subsample=(2, 2), activation = 'relu'))
model.add(Conv2D(64, kernel_size=3, activation = 'relu'))
model.add(Conv2D(64, kernel_size=3, activation = 'relu'))
#model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

'''
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x /255.0 - 0.5))
model.add(Conv2D(16, kernel_size=3, subsample=(2, 2), activation = 'relu'))
model.add(Dropout(0.4))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1))
'''

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=3, validation_split=0.2)

model.save('model.h5')
          