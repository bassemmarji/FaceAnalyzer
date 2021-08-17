#Import Libraries
import dlib,cv2,imutils,os,argparse,filetype
from imutils import face_utils

#Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
#in a person's face
FACIAL_LANDMARK_PREDICTOR = '.\\static\\models\\shape_predictor_68_face_landmarks.dat'

#The gender model architecture
GENDER_MODEL = '.\\static\\models\\gender_deploy.prototxt'
#The gender model pre-trained weights
GENDER_PROTO = '.\\static\\models\\gender_net.caffemodel'
#Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
#substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#Represent the gender classes
GENDER_LIST = ['Male', 'Female']

def initialize_dlib(facial_landmark_predictor:str):
    """
    #Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor

def load_caffe_models(gender_model:str,gender_proto:str):
    """
    load the pre-trained Caffe model for gender estimation
    """
    gender_net    = cv2.dnn.readNetFromCaffe(gender_model , gender_proto)
    #use CPU
    gender_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    return gender_net

def display_img(title,img):
    """
    Displays an image on screen and maintains the output until the user presses a key
    """
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowTitle('img',title)
    cv2.resizeWindow('img',600,400)

    #Display Image on screen
    cv2.imshow('img',img)

    #Mantain output until user presses a key
    cv2.waitKey(0)

    #Destroy windows when user presses a key
    cv2.destroyAllWindows()

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

def predict_gender(input_path:str):
    """
    Predict the gender of the faces showing in the image
    """
    #Initialize dlib face detector using the facial landmark recognition
    detector, predictor = initialize_dlib(facial_landmark_predictor=FACIAL_LANDMARK_PREDICTOR)

    #Load age prediction model
    gender_net = load_caffe_models(gender_model=GENDER_MODEL, gender_proto = GENDER_PROTO)

    # Initialize frame size
    frame_width = 640
    frame_height = 360

    # Read Input Image
    img = cv2.imread(input_path)

    #Copy the initial image and resize it
    frame = img.copy()
    frame = imutils.resize(img, width=frame_width, height=frame_height)

    # Convert it to gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(gray_frame, 0)

    # Loop over the faces detected
    for idx, face in enumerate(faces):
        print("Detection Face ID = {} - Position = Left:{} Top:{} Right:{} Botton:{}".format((idx+1), face.left(), face.top(), face.right(),
                                                                       face.bottom()))
        #Draw the face bounding box
        (x,y,w,h) = face_utils.rect_to_bb(face)
        startX , startY , endX , endY = x,y,(x+w),(y+h)
        face_img = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        #Retrieve face
        # Determine the facial landmarks for the face region
        #shape = predictor(gray_frame, face)
        # Convert the facial landmark (x, y) coordinates to a NumPy array
        #shape = face_utils.shape_to_np(shape)
        # Extract the ROI of the face region as a separate image
        #(x, y, w, h) = cv2.boundingRect(np.array([shape]))
        #roi = img[y:y + h, x:x + w]
        #display_img("face", roi)

        # image --> Input image to preprocess before passing it through our dnn for classification.
        #scale factor = After performing mean substraction we can optionally scale the image by some factor. (if 1 -> no scaling)
        #size = The spatial size that the CNN expects. Options are = (224*224, 227*227 or 299*299)
        #mean = mean substraction values to be substracted from every channel of the image.
        #swapRB=OpenCV assumes images in BGR whereas the mean is supplied in RGB. To resolve this we set swapRB to True.
        blob = cv2.dnn.blobFromImage(image= face_img
                                     , scalefactor=1.0
                                     , size=(227, 227)
                                     , mean=MODEL_MEAN_VALUES
                                     , swapRB=False
                                     , crop=False)
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]

        #Draw the box
        label = "{}-{:.2f}%".format(gender,gender_confidence_score*100)
        print(label)

        yPos = startY - 15
        while yPos < 15:
              yPos +=  15

        optimal_font_scale = get_optimal_font_scale(label,((endX-startX)+25))

        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(face_img, (startX, startY), (endX, endY), box_color, 2)
        #Label processed image
        cv2.putText(face_img, label, (startX, yPos), cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale , box_color , 2)

        # Display processed image
        display_img("Gender Estimator", face_img)

    # Cleanup
    cv2.destroyAllWindows()


def is_valid_path(path):
    """
    Validates the path inputted and makes sure that it is a file of type image
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
    predict_gender(input_path  = args['input_path'])
