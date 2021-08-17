#Import Libraries
import dlib,cv2,imutils,os,argparse,filetype
from imutils import face_utils
import numpy as np
from PIL import Image, ImageDraw, ImageFont

#Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
#in a person's face
FACIAL_LANDMARK_PREDICTOR = '.\\static\\models\\shape_predictor_68_face_landmarks.dat'

def initialize_dlib(facial_landmark_predictor:str):
    """
    #Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor

def get_emoji(emotion):
    """
    Return the emoji corresponding to the emotional state
    """
    import emoji
    return {
      'angry':emoji.emojize(":angry_face:")
     ,'disgust':emoji.emojize(":weary_face:")
     ,'fear': emoji.emojize(":worried_face:")
     ,'happy': emoji.emojize(":smiling_face:")
     ,'sad': emoji.emojize(":frowning_face:")
     ,'surprise': emoji.emojize(":hushed_face:")
     ,'neutral': emoji.emojize(":neutral_face:")
    }[emotion]

def layout_emotions(emotions):
    """
    Layout emotions in an image
    """
    emotions_img = Image.new('RGB',(300,300),color=(0,0,0,0))
    #Get a drawing context
    draw = ImageDraw.Draw(emotions_img)
    font = ImageFont.truetype(".\\static\\models\\OpenSansEmoji.ttf",30,encoding='unic')

    for idx, (emotion, score) in enumerate(emotions.items()):
        if emotion:
            score = 0 if score is None else score
            label = "{}-{}-{:.2f}%".format(get_emoji(emotion),emotion,score * 100)
            draw.text((20,40 * idx),label,font=font,embedded_color=True)

    emotions_img = np.array(emotions_img)
    return emotions_img

def get_emotion(img):
    """
    Gather emotions
    """
    from fer import FER
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(img)
    emotion, emotion_score = detector.top_emotion(img)
    emotion_score = 0 if emotion_score is None else emotion_score
    top_emotion = "Top emotion: {} - {:.2f}%".format(emotion, emotion_score * 100)
    print(top_emotion)
    emotions_layout = None
    if result:
        emotions = result[0]["emotions"]
        emotions_layout = layout_emotions(emotions)
    return top_emotion, emotions_layout

def predict_emotion(input_path:str):
    """
    Predict the emotion of the faces showing in the image
    """
    #Initialize dlib face detector using the facial landmark recognition
    detector, predictor = initialize_dlib(facial_landmark_predictor=FACIAL_LANDMARK_PREDICTOR)

    # Read Input Image
    img = cv2.imread(input_path)

    # Resize it
    frame = img.copy()

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

        #Retrieve face
        # Determine the facial landmarks for the face region
        shape = predictor(gray_frame, face)
        # Convert the facial landmark (x, y) coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)
        # Extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape]))
        roi = img[y:y + h, x:x + w]

        top_emotion, emotions_layout = get_emotion(img = roi)

        combined_img = cv2.resize(roi , (400, 400), cv2.INTER_CUBIC)
        if emotions_layout is not None:
           combined_img = cv2.hconcat([cv2.resize(roi , (400, 400), cv2.INTER_CUBIC)
                                      ,cv2.resize(emotions_layout, (400, 400), cv2.INTER_CUBIC)])

        # Display Image on screen
        cv2.imshow(top_emotion,combined_img)
        # Mantain output until user presses a key
        cv2.waitKey(0)

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
    predict_emotion(input_path  = args['input_path'])
