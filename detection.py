# Python program to implement Breath detection algorithm

# importing library 
import cv2, time, pandas, winsound, pyttsx3, smtplib
from email.message import EmailMessage
from datetime import datetime
import datetime as dt

# Assigning our static_background to None 
static_back = None
tracker = cv2.TrackerCSRT_create()

# List when any moving object appear or not
motion_list = [ None, None ] 
iterations=1 
current=None

# Time of movement 
time = [] 

# appending frame
fm=[]  

# Initializing DataFrame, one column is start time and other column is end time 
df = pandas.DataFrame(columns = ["Start", "End"]) 
  
# Capturing video 
video = cv2.VideoCapture(0) 

frm = video.read()[1]
roi=cv2.selectROI("Tracking",frm)

tracker.init(frm,roi)
# Infinite while loop to treat stack of image as video 
while True:
    
    # Reading frame(image) from video 
    check, frame = video.read()

    if check==True:
        # Initializing motion = 0(no motion)
        success, r=tracker.update(frame)
        motion = 0
        
        if success:
            frame=frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        else:
            frame=frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Converting gray scale image to GaussianBlur, so that change can be find easily 
        gray = cv2.GaussianBlur(gray, (21, 21), 0)  
        cv2.imshow("Gaussian Blur", gray)
        # Converting GaussianBlur to canny edge
        edge = cv2.Canny(gray, 34, 34)
        gray = cv2.GaussianBlur(edge, (21, 21), 0)
        
        fm.append(gray)

        # In first iteration we assign the value of static_back to our first frame 
        if static_back is None: 
            static_back = gray
            for i in range(0,16,1):
                fm.append(gray)
        else:
            static_back=fm[-16]

        # Difference between static background and current frame (which are GaussianBlur frames) 
        try:
            diff_frame = cv2.absdiff(static_back, gray)
        except:
            continue

        # If change in between static background and current frame is greater than 30 it will show white color(255) 
        thresh_frame = cv2.threshold(diff_frame, 10, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 10)

        # Finding contour of moving object 
        cnts = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in cnts:
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour) 
            # making green rectangle arround the moving object 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

        if motion==0:
            winsound.Beep(4000,25)
            
            if iterations==1:
                current=datetime.now().replace(microsecond=0)+dt.timedelta(seconds=10)

            if current==datetime.now().replace(microsecond=0):

                # voice alert system
                engine = pyttsx3.init()
                engine.say("Found no motion. Please check immediately")
                engine.runAndWait()

                # mail alert system
                msg = EmailMessage()
                mail = smtplib.SMTP('smtp.gmail.com', 587)
                mail.starttls()
                mail.login("from_mail@gmail.com","password")
                sub="Motion not detected"
                body="The patient is not moving. Please check immediately."
                msg=f'Subject: {sub} \n {body}'
                mail.sendmail("from_mail@gmail.com", "your_email@gmail.com", msg) 

            iterations+=1
        else:
            iterations=1

        # Appending status of motion 
        motion_list.append(motion) 
                        
        # Appending Start time of no-motion 
        if motion_list[-1] == 0 and motion_list[-2] == 1: 
            time.append(datetime.now()) 

        # Appending End time of no-motion 
        if motion_list[-1] == 1 and motion_list[-2] == 0: 
            time.append(datetime.now())

        # Displaying image in gray_scale 
        #cv2.imshow("Gaussian Blur", gray) 

        # Displaying the difference in currentframe to the staticframe 
        cv2.imshow("Difference Frame", diff_frame) 

        # Displaying the black and white image in which if intensity difference greater than 30 it will appear white 
        cv2.imshow("Threshold Frame", thresh_frame) 

        # Displaying color frame with contour of motion of object 
        cv2.imshow("Color Frame", frame) 

        # Displaying canny edge frame
        cv2.imshow('Canny Edge', edge)

        key = cv2.waitKey(50) 
        
        # if q entered whole process will stop 
        if key == ord('q'):
            # if something is moving then it append the end time of movement 
            if motion == 0: 
                time.append(datetime.now())
            break

    else:
        print("The video has ended or cannot be converted into frame")
        break


# Appending time when no-motion is observed in DataFrame 
for i in range(1, len(time), 2): 
    df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True) 
  
# Creating a CSV file in which time no-movements occurred will be saved 
df.to_csv("Time_of_no-movements.csv")

video.release()

# Destroying all the windows 
cv2.destroyAllWindows()
