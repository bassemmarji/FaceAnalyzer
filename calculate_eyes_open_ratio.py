#Import Libraries
import dlib,cv2,os,argparse,filetype
from imutils import face_utils
from scipy.spatial import distance

#Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
#in a person's face
FACIAL_LANDMARK_PREDICTOR = '.\\static\\models\\shape_predictor_68_face_landmarks.dat'

#Minimum ratio determining if eyes are closed
EYE_OPEN_RATIO = 0.2

def initialize_dlib(facial_landmark_predictor:str):
    """
    Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor

def calc_eye_open_ratio(eye):
    """
    Calculate the eye open ratio
    """
    #Compute the euclidean distance between the two sets of vertical eye coordinates
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    # Compute the euclidean distance between the two sets of horizontal eye coordinates
    C = distance.euclidean(eye[0], eye[3])

    # Calculate the eye aspect ratio
    avgor = ((A+B) / (2.0*C))
    eor = 0
    #Eye is not blinking
    if avgor >  EYE_OPEN_RATIO:
       #Calculating Eye Open Ratio
       eor = 1 - ((A+B) / (A*B))

    #print('A',A,'B',B,'C',C,'avgor',avgor,'eor',eor)
    #return the eye open ration
    return eor


def get_optimal_font_scale(text, width):
    """
    Determine the optimal font scale based on the hosting frame width
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        #print(new_width)
        if (new_width <= width):
            return scale/10
    return 1

def calculate_eyes_open_ratio(input_path:str):
    """
    Calculate the eyes open ratio of the faces in the image showing in the input path
    """
    #Initialize dlib face detector using the facial landmark recognition
    detector, predictor = initialize_dlib(facial_landmark_predictor=FACIAL_LANDMARK_PREDICTOR)

    # Read Input Image
    img = cv2.imread(input_path)

    # Copy the original image
    frame = img.copy()

    # Convert it to gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(gray_frame, 0)

    #Left eye
    (left_Start, left_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    #Right eye
    (right_Start, right_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    eyes = []

    # Loop over the faces detected
    for idx, face in enumerate(faces):
        print("Detection Face ID = {} - Position = Left:{} Top:{} Right:{} Botton:{}".format((idx+1), face.left(), face.top(), face.right(),
                                                                       face.bottom()))

        #Draw the face bounding box
        (x,y,w,h) = face_utils.rect_to_bb(face)
        #face_img = frame[y:y+h,x:x+w]

        #Draws blue box over the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0),3)

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y) coordinates to a NumPy array
        shape = predictor(gray_frame, face)
        shape = face_utils.shape_to_np(shape)

        # indexes for left eye key points
        leftEye = shape[left_Start:left_End]
        rightEye = shape[right_Start:right_End]

        leftEOR  = calc_eye_open_ratio(leftEye)
        rightEOR = calc_eye_open_ratio(rightEye)
        #Avg Eye Open Ration for both eyes
        avg_eor = (leftEOR + rightEOR) / 2.0

        #wrap eyes in a list
        eyes.append(leftEye)
        eyes.append(rightEye)

        #Localize eyes
        for index, eye in enumerate(eyes):
            #Left edge of the eye
            left_side_eye = eye[0]
            #Right edge of the eye
            right_side_eye = eye[3]
            #Top side of the eye
            top_side_eye = eye[1]
            #Bottom side of the eye
            bottom_side_eye = eye[4]

            #calculate height and width of dlib eye keypoints
            eye_width = right_side_eye[0] - left_side_eye[0]
            eye_height = bottom_side_eye[1] - top_side_eye[1]

            # create bounding box with buffer around keypoints
            eye_x1 = int(left_side_eye[0]  - 0 * eye_width)
            eye_x2 = int(right_side_eye[0] + 0 * eye_width)
            eye_y1 = int(top_side_eye[1]   - 1 * eye_height)
            eye_y2 = int(bottom_side_eye[1]+ 0.75  * eye_height)

            startX, startY, endX, endY =  eye_x1, eye_y1, eye_x2, eye_y2

            if index == 0:
                label = "{:.2f}%".format(leftEOR * 100)
            elif index == 1:
                label = "{:.2f}%".format(rightEOR * 100)

            yPos = startY - 15
            while yPos < 15:
                yPos += 15

            optimal_font_scale = get_optimal_font_scale(label, ((endX - startX) + 25))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, yPos), cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, (0, 255, 0), 2)

        title = "Left Eye Open Ratio = {:.2f}%".format(leftEOR*100)
        title += " --- Right Eye Open Ratio = {:.2f}%".format(rightEOR * 100)
        title += " --- Average Eyes Open Ratio = {:.2f}%".format(avg_eor * 100)
        #Concatenate images
        combined_img = cv2.hconcat([cv2.resize(img   , (400, 400), cv2.INTER_CUBIC)
                                   ,cv2.resize(frame , (400, 400), cv2.INTER_CUBIC)])

        # Display Image on screen
        cv2.imshow(title,combined_img)
        # Mantain output until user presses a key
        cv2.waitKey(0)

    # Cleanup
    cv2.destroyAllWindows()

def is_valid_path(path):
    """
    Validates the path inputted and validates that it is a file of type image
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
    calculate_eyes_open_ratio(input_path  = args['input_path'])
