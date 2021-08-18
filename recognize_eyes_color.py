#Import Libraries
import dlib,cv2,os,argparse,filetype
from imutils import face_utils
import webcolors

#Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
#in a person's face
FACIAL_LANDMARK_PREDICTOR = '.\\static\\models\\shape_predictor_68_face_landmarks.dat'

def initialize_dlib(facial_landmark_predictor:str):
    """
    Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor

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
    #Initialize dlib face detector using the facial landmark recognition
    detector, predictor = initialize_dlib(facial_landmark_predictor=FACIAL_LANDMARK_PREDICTOR)

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert it to gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(gray_frame, 0)

    #Left eye
    (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    #Right eye
    (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Loop over the faces detected
    for idx, face in enumerate(faces):
        print("Detection Face ID = {} - Position = Left:{} Top:{} Right:{} Botton:{}".format((idx+1), face.left(), face.top(), face.right(),
                                                                       face.bottom()))
        #Draw the face bounding box
        (x,y,w,h) = face_utils.rect_to_bb(face)
        #face_img = frame[y:y+h,x:x+w]

        #Store the 2 eyes
        eyes = []

        #Draws blue box over the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0),3)

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y) coordinates to a NumPy array
        shape = predictor(gray_frame, face)
        shape = face_utils.shape_to_np(shape)

        # indexes for left eye key points
        leftEye = shape[left_Start:left_End]
        rightEye = shape[right_Start:right_End]

        # wrap in a list
        eyes.append(leftEye)
        eyes.append(rightEye)
        flag = 0

        for index, eye in enumerate(eyes):
            flag += 1
            #Left edge of the eye
            left_side_eye = eye[0]
            #Right edge of the eye
            right_side_eye = eye[3]
            #Top side of the eye
            top_side_eye = eye[1]
            #Bottom side of the eye
            bottom_side_eye = eye[4]

            # calculate height and width of dlib eye keypoints
            eye_width = right_side_eye[0] - left_side_eye[0]
            eye_height = bottom_side_eye[1] - top_side_eye[1]

            # create bounding box with buffer around keypoints
            eye_x1 = int(left_side_eye[0]  - 0 * eye_width)
            eye_x2 = int(right_side_eye[0] + 0 * eye_width)
            eye_y1 = int(top_side_eye[1]   - 1 * eye_height)
            eye_y2 = int(bottom_side_eye[1]+ 0.75  * eye_height)

            # draw bounding box around eye roi
            rect = cv2.rectangle(frame,(eye_x1, eye_y1), (eye_x2, eye_y2),(255,0,0),2)

            # Get the desired eye region (RGB)
            roi_eye = frame[eye_y1:eye_y2, eye_x1:eye_x2]

            if flag==1:
               break

        x = roi_eye.shape
        row = x[0]
        col = x[1]

        #Pick RGB values from the area just below pupil
        arr_eye_color = roi_eye[row //2:(row //2) +1,((col //3) +3):((col //3)) + 6]
        array1 = arr_eye_color[0][2]

        # store it in tuple and pass this tuple to "find_color" Funtion
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

