import cv2
from utils.SORT import Tracking

if __name__ == "__main__":
    t = Tracking()
    video_path = './test_video/videoplayback.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Loop to read and display frames
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        '''
        tracking return : -> XYWH 
        
        '''
        test = t.detect(frame)

        height,width,ch = frame.shape

        # # Resize the image
        # frame = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))

        # height,width,ch = frame.shape

        class_name = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Bicycle']
        class_color = [(0, 0, 255) , (221,160,221) , (80,127,255), ( 0,215,255), (203,192,255)]
        # If frame is read correctly, ret is True
        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        

        len_item = len(test[list(test.keys())[0]])
        for i in range(len_item):

            x, y, w, h = test['xywh'][i]
            ID = test['ID'][i]
            x_min = int((x - w / 2) * width)
            y_min = int((y - h / 2) * height)
            x_max = int((x + w / 2) * width)
            y_max = int((y + h / 2) * height)
            conf = test['conf'][i]
            cls = test['cls'][i]

            overlay = frame.copy()

            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), class_color[int(cls)], -1)
            cv2.addWeighted(overlay, 0.4, frame, 1 - 0.4, 0, frame)
            text_x, text_y = x_min, y_min - 20  # Offset the text above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            
            color = (0, 255, 0)  # Green color
            thickness = 2
            cv2.putText(frame, "ID : " + str(ID), (text_x, text_y), font, font_scale, color, thickness)
            cv2.putText(frame, str([class_name[int(cls)]]), (text_x, text_y + 13)
                        , font, font_scale, (255, 0, 0), thickness)

        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    # Close all OpenCV windows
