#Import Libraries
import cv2,os,argparse,filetype
import webcolors,imutils

#The path to the pre-trained Haar cascade face detector
HAAR_CASCADE_FACE_DETECTOR_PATH = '.\\static\\cascades\\haarcascade_frontalface_default.xml'
#The path to the pre-trained Haar cascade eyes detector
HAAR_CASCADE_EYES_DETECTOR_PATH = '.\\static\\cascades\\haarcascade_eye.xml'

def initialize_cascade_detectors():
    """
    Initialize and the pre-trained Haar cascade detectors
    """
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FACE_DETECTOR_PATH)
    eye_cascade = cv2.CascadeClassifier(HAAR_CASCADE_EYES_DETECTOR_PATH)
    return face_cascade, eye_cascade

def closest_color(req_color):
    """
    Convert an RGB pixel to a color name
    """
    min_colours = {}
    for key,name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - req_color[0]) ** 2
        gd = (g_c - req_color[1]) ** 2
        bd = (b_c - req_color[2]) ** 2
        min_colours[(rd + gd + bd)] = name
        closest_name = min_colours[min(min_colours.keys())]
    return closest_name


def find_color(req_color):
    """
    Find the actual color, if not the closest one
    """
    try:
        closest_name = actual_name = webcolors.rgb_to_name(req_color)
    except ValueError:
        closest_name = closest_color(req_color)
        actual_name  = None

    result = actual_name if actual_name else closest_name
    return result

def recognize_eyes_color(input_path:str):
    """
    Recognize the color of the eyes of the faces showing within the image
    """
    #Initialize the Haar cascade face recognizer
    face_cascade,eye_cascade = initialize_cascade_detectors()

    # Read Input Image
    img = cv2.imread(input_path)

    # Initialize frame size
    frame_width = 640
    frame_height = 360

    # Preserve a copy of the original
    frame = img.copy()
    frame = imutils.resize(frame, width=frame_width, height=frame_height)

    # Convert it to gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = face_cascade.detectMultiScale(
                                          #Matrix of type CV_8U containing objects to be detected
                                          image=gray_frame
                                          #Specifying how much the image will be reduced at each scale
                                        , scaleFactor=1.05
                                          #Affect the quality of detected objects
                                          #Higher values result in less detections but with higher quality
                                        , minNeighbors=5
                                          #Minimum possible object size. Objects smaller are ignored
                                        ,minSize=(30,30)
                                        ,flags=cv2.CASCADE_SCALE_IMAGE
                                        )

    print("{} face(s) detected".format(len(faces)))

    # Loop over the faces detected
    for idx, (x,y,w,h) in enumerate(faces):
        print("Detection Face ID = {}".format((idx+1)))

        #Draws blue box over the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0),3)

        #Region of ineterest in gray mode
        roi_face_gray  = gray_frame[y:y+h,x:x+w]
        # Region of ineterest in color mode
        roi_face_color = frame[y:y+h,x:x+w]

        #Detecting eyes
        eyes = eye_cascade.detectMultiScale(image=roi_face_gray
                                          , scaleFactor = 1.3
                                          # Affect the quality of detected objects
                                          # Higher values result in less detections but with higher quality
                                          , minNeighbors = 9
                                          # Minimum possible object size. Objects smaller are ignored
                                          , minSize = (30, 30))

        print("{} eye(s) detected".format(len(eyes)))

        roi_eye = None
        flag = 0
        for idx, (x_eye, y_eye,eye_width,eye_height) in enumerate(eyes):

            flag += 1
            #Left edge of the eye
            left_side_eye = x_eye
            #Right edge of the eye
            right_side_eye = x_eye + eye_width
            #Top side of the eye
            top_side_eye = y_eye + eye_height
            #Bottom side of the eye
            bottom_side_eye = y_eye

            # create bounding box with buffer around keypoints
            eye_x1 = int(left_side_eye   - 0     * eye_width)
            eye_x2 = int(right_side_eye  + 0.2   * eye_width)
            eye_y1 = int(top_side_eye    - 0.75  * eye_height)
            eye_y2 = int(bottom_side_eye + 0.75  * eye_height)

            # draw bounding box around eye roi
            rect = cv2.rectangle(roi_face_color,(eye_x1, eye_y1), (eye_x2, eye_y2),(255,0,0),2)

            # Get the desired eye region (RGB)
            roi_eye = roi_face_color[eye_y1:eye_y2, eye_x1:eye_x2]

            if flag==2:
               break

        if roi_eye is not None:
           x = roi_eye.shape
           row = x[0]
           col = x[1]

           #Pick RGB values from the area just below pupil
           arr_eye_color = roi_eye[row //2:(row //2) +1,((col //3) +3):((col //3)) + 6]

           array1 = arr_eye_color[0][2]

           #store it in tuple and pass this tuple to "find_color" Funtion
           label = "Eyes color = {}".format( str(find_color(array1)).title())

           combined_img = cv2.hconcat([cv2.resize(frame        , (200, 200), cv2.INTER_CUBIC)
                                      ,cv2.resize(arr_eye_color, (200, 200), cv2.INTER_CUBIC)])
           if arr_eye_color is not None:
               combined_img = cv2.hconcat([cv2.resize(frame        , (200, 200) , cv2.INTER_CUBIC)
                                          ,cv2.resize(roi_eye      , (200, 200) , cv2.INTER_CUBIC)
                                          ,cv2.resize(arr_eye_color, (200, 200) , cv2.INTER_CUBIC)])

           # Display Image on screen
           cv2.imshow(label,combined_img)
           # Mantain output until user presses a key
           cv2.waitKey(0)

    # Cleanup
    cv2.destroyAllWindows()


def is_valid_path(path):
    """
    Validates the path inputted and makes sure that is a file of type image
    """
    if not path:
        raise ValueError(f"Invalid Path")
    if os.path.isfile(path) and 'image' in filetype.guess(path).mime:
       return path
    else:
       raise ValueError(f"Invalid Path {path}")


def parse_args():
    """
    Get user command line parameters
    """
    parser = argparse.ArgumentParser(description="Available Options")

    parser.add_argument('-i'
                       ,'--input_path'
                       ,dest='input_path'
                       ,type=is_valid_path
                       ,required=True
                       ,help = "Enter the path of the image file to process")

    args = vars(parser.parse_args())

    #To Display The Command Line Arguments
    print("## Command Arguments #################################################")
    print("\n".join("{}:{}".format(i,j) for i,j in args.items()))
    print("######################################################################")

    return args

if __name__ == '__main__':
    # Parsing command line arguments entered by user
    args = parse_args()
    recognize_eyes_color(input_path  = args['input_path'])

