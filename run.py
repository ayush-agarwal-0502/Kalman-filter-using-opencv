#kalman filter implementation in opencv
#made by AYUSH AGARWAL , Electronics engg , IIT BHU Varanasi
#importing the libraries
import cv2
import numpy as np
import math
import random
#getting the video camera input
capture = cv2.VideoCapture(0)
#######################################################
#Ayush Agarwal ECE
#making the class for kalman filter with modifications suited to my project
class kalman:
    #made a class for kalman cause the online ones are only for older version
    def __init__(self):
        #state contains old and present position , just like the q they asked me in roboreg interview
        #to make the process marlov , we can take present and past state as a combined state
        self.state = [0,0,0,0] #first 2 elements are from t-1 and other 2 are for t
        self.state_transition_matrix = [1]
        #no control matrix needed for my task , just wrote to clear it
        self.control_matrix = [0,0]
        self.control_factor = [0]
    def predict(self):
        #here I have made the prediction that object moves in the direction it was moving in and hence got the predicted position
        #in real kalman filter prediction must be a linear function of present state and control(which isnt here)
        #hence this is also another way of writing the kalman filter
        #i've added gaussian noise too , and updated the state properly
        x_tf = self.state[2] + (self.state[2] - self.state[0]) + random.gauss(0,10**(-5))
        y_tf = self.state[3] + (self.state[3] - self.state[1]) + random.gauss(0,10**(-5))
        self.state[0] = self.state[2]
        self.state[1] = self.state[3]
        self.state[2] = x_tf
        self.state[3] = y_tf
        #now state has first 2 elements as the position from which calc started and last 2 as the predicted coordinates
        #the actual kalman filter is supposed to return the predicted state but im only returning half as per convenience
        return [x_tf,y_tf]
    def correct(self,measurement_matrix):
        #measurement matrix will contain the position that opencv would have detected of the object
        #I've simply taken the average of the predicted position and measured position for the correction part
        x_cor = (self.state[2] + measurement_matrix[0])/2 + random.gauss(0,10**(-5))
        y_cor = (self.state[3] + measurement_matrix[1])/2 + random.gauss(0,10**(-5))
        #now first 2 of state contain the predicted position and last 2 contain the corrected position
        self.state[0] = self.state[2]
        self.state[1] = self.state[3]
        self.state[2] = x_cor
        self.state[3] = y_cor
        return [x_cor,y_cor]

#######################################################
def image_processing_mask_prep(frame):
    #converting to hsv for masking
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #green colour code in hsv
    lower_colour = np.array([40,100,100])
    upper_colour = np.array([80,255,255])
    #now creating the mask
    #inrange is just the hsv way of thresholding in opencv
    mask1 = cv2.inRange(hsv,lower_colour,upper_colour)
    #kernel for morphological operations, different kernels for different operations
    kernel = np.ones((5,5),np.uint8)
    kernel1 = np.ones((25,25),np.uint8)
    kernel2 = np.ones((1,1),np.uint8)
    #dilating to spread the small detected area
    mask1 = cv2.dilate(mask1,kernel1)
    #open removes false positives in background
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,kernel)
    #closing removes false negatives in object
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernel1)
    #blurring to remove the dots further
    mask1 = cv2.GaussianBlur(mask1,(5,5),0)
    #finally eroding to give it a good shape and feel at edges
    mask1 = cv2.erode(mask1,kernel2)
    #returning the prepared mask
    return mask1

#######################################################
#creating the instance of the class
kalu = kalman()
#standard opencv procedure for videos
while True:
    ret, frame = capture.read()
    if (ret==True):
        #flipping for making it like a mirror
        frame = cv2.flip(frame,100)
        #total width and height of video
        W,H,_ = frame.shape
        #cv2.imshow("video frame 1",frame)
        #getting the mask from the function
        mask1 = image_processing_mask_prep(frame)
        cv2.imshow("green mask",mask1)
        cv2.waitKey(1)
        #now getting the contours in the prepared mask 
        contours,_ = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #now drawing a rectangle around the detected object
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            cv2.putText(frame,"The original detected object ",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        #sometimes the code bugs so uncomment this line if it shows error that x is unknown
        #however if this line is uncommented , make sure that u are holding a green object in front of the camera
        #otherwise the program will end automatically
        #x, y, w, h = cv2.boundingRect(contours[0])

        #getting the predicted coordinates from the class
        predicted_coords = kalu.predict()
        #drawing a red circle around the predicted position and writing over it
        cv2.circle(frame,(int(predicted_coords[0]),int(predicted_coords[1])),25,[0,0,255],2)
        cv2.putText(frame,"Predicted position ",(int(predicted_coords[0]),int(predicted_coords[1])-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        #getting the corrected coordinates from the kalman
        corrected_coord = kalu.correct([(x+(w/2)),(y+(h/2))])
        #drawing a circle and writing text around the corrected kalman prediction
        cv2.circle(frame,(int(corrected_coord[0]),int(corrected_coord[1])),25,[255,0,0],2)
        cv2.putText(frame,"Corrected position ",(int(corrected_coord[0]),int(corrected_coord[1])-40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        #showing the frame after putting all the things
        cv2.imshow("video frame 1",frame)
        if (cv2.waitKey(1)==ord("q")):
            break
    else:
        print("Error in loading the window")
#standard opencv way to close the window
capture.release()
cv2.destroyAllWindows()

