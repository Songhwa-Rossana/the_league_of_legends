import os
import argparse

import numpy as np
import cv2 as cv

import opencv_zoo.models as cv_models # TODO Put this more correct
from utils import palm_rots, get_palm_params, prompt_text_user
from skimage.feature import hog

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser()
parser.add_argument('--database_dir', '-db', type=str, default='./database')
parser.add_argument('--palm_detection_model', '-fd', type=str, default='./palm_detection_mediapipe_2023feb.onnx')
#parser.add_argument('--face_recognition_model', '-fr', type=str, required=True)
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--score_threshold', type=float, default=0.8,
                    help='Usage: Set the minimum needed confidence for the model to identify a palm, defaults to 0.8. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold. An empirical score threshold for the quantized model is 0.49.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--rot_threshold', type=float, default=0.2,
                    help='Usage: Maximum hand rotation allowed (radians). Default = 0.2.')
args = parser.parse_args()

def detect_palm(detector, image):
    ''' Run face detection on input image.

    Paramters:
        detector - an instance of cv.MPPalmDet
        image    - a single image read using cv.imread

    Returns:
        faces    - a np.array of shape [n, 15]. If n = 0, return an empty list.
    '''
    palms = []
    ### TODO: your code starts here
    palms = detector.infer(image) # [x1, y1, x2, y2, landmarks...]
    if palms.size > 0:
        palms_landmarks = palms[:, 4:-1].reshape(len(palms), -1, 2)
        rot = palm_rots(palms_landmarks)
        palms = np.concatenate((palms, rot[..., np.newaxis]), axis=1)
    ### your code ends here
    return palms

def extract_feature(recognizer, image, palms):
    ''' Run face alignment on the input image & face bounding boxes; Extract features from the aligned faces.

    Parameters:
        recognizer - an instance of MPPalmDet
        image      - a single image read using cv.imread
        palms      - the return of detect_palm

    Returns:
        features   - a length-n list of extracted features. If n = 0, return an empty list.
    '''
    features = []
    ### TODO: your code starts here
    for palm in palms:

        # Isolate the image region containing the palm
        bbox, landmarks, _ = get_palm_params(palm)
        palm_image = image[bbox[0]: bbox[2] + 1, bbox[1]: bbox[3] + 1]
        if palm_image.size == 0:
            continue

        palm_image_gray = cv.cvtColor(palm_image, cv.COLOR_BGR2GRAY)
        palm_image_gray = palm_image_gray.astype(np.float32) / 255.0
        palm_image_gray = cv.resize(palm_image_gray, recognizer.winSize)
        #feature = recognizer.compute(palm_image_gray)
        feature = hog(palm_image_gray, block_norm='L2-Hys', pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=False)
        features.append(feature)
    # TODO Check if is valid
    ### your code ends here
    return np.array(features)

def match(recognizer, feature1, feature2, dis_type=1):
    ''' Calculate the distatnce/similarity of the given feature1 and feature2.

    Parameters:
        recognizer - an instance of cv.FaceRecognizerSF. Call recognizer.match to calculate distance/similarity
        feature1   - extracted feature from identity 1
        feature2   - extracted feature from identity 2
        dis_type   - 0: cosine similarity; 1: l2 distance; others invalid

    Returns:
        isMatched  - True if feature1 and feature2 are the same identity; False if different
    '''
    l2_threshold = 10 #1.128
    cosine_threshold = 0.363
    isMatched = False
    ### TODO: your code starts here
    if dis_type == 0:  # cosine
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        if dist < cosine_threshold:
            isMatched = True
    elif dis_type == 1: # l2 distance
        dist = np.linalg.norm(feature1 - feature2, ord=2)
        if dist < l2_threshold:
            isMatched = True
    else:
        raise ValueError(f'{dis_type} is not valid')
    ### your code ends here
    return isMatched, dist

def load_database(database_path, detector, recognizer):
    ''' Load database from the given database_path into a dictionary. It tries to load extracted features first, and call detect_palm & extract_feature to get features from images (*.jpg, *.png).

    Parameters:
        database_path - path to the database directory
        detector      - an instance of cv.FaceDetectorYN
        recognizer    - an instance of cv.FaceRecognizerSF

    Returns:
        db_features   - a dictionary with filenames as key and features as values. Keys are used as identity.
    '''
    db_features = dict()

    print('Loading database ...')
    # load pre-extracted features first
    for filename in os.listdir(database_path):
        if filename.endswith('.npy'):
            identity = filename[:-4]
            if identity not in db_features:
                db_features[identity] = np.load(os.path.join(database_path, filename))
    npy_cnt = len(db_features)
    # load images and extract features
    for filename in os.listdir(database_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            identity = filename[:-4]
            if identity not in db_features:
                image = cv.imread(os.path.join(database_path, filename))
                palms = detect_palm(detector, image)
                features = extract_feature(recognizer, image, palms)
                if len(features) > 0:
                    db_features[identity] = features[0]
                    np.save(os.path.join(database_path, '{}.npy'.format(identity)), features[0])
    cnt = len(db_features)
    print('Database: {} loaded in total, {} loaded from .npy, {} loaded from images.'.format(cnt, npy_cnt, cnt-npy_cnt))
    return db_features

def visualize(image, palms, identities, valids, fps, valid_box_color=(0, 255, 0), non_valid_box_color=(0, 165, 255), text_color=(0, 0, 255)):
    output = image.copy()

    # put fps in top-left corner
    cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    for palm, identity, valid in zip(palms, identities, valids):

        # draw bounding box
        bbox, palm_landmarks, palm_rot = get_palm_params(palm)
        #bbox = palm[0:4].astype(np.int32)

        if not valid:
            cv.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), non_valid_box_color, 2)
            # Prompt the user to put the hand toward the camera
            cv.putText(output, 'Put hand vertically', (bbox[0], bbox[1]-15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        else:
            cv.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), valid_box_color, 2)
            # put identity
            cv.putText(output, '{}'.format(identity), (bbox[0], bbox[1]-15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        # draw points
        #cv.line(output, palm_landmarks[1], palm_landmarks[4], (0, 0, 255), 2) # TODO Add a color
        #for p in palm_landmarks:
        #    cv.circle(output, p, 2, (0, 0, 255), 2) # TODO Add a color

        # Compute stats
        #cv.putText(output, 'rot: {}'.format(palm_rot), (bbox[0], bbox[1]-15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    return output

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Initialize MPPalmDet
    detector = cv_models.MPPalmDet(modelPath=args.palm_detection_model,
                      nmsThreshold=args.nms_threshold,
                      scoreThreshold=args.score_threshold,
                      backendId=backend_id,
                      targetId=target_id)
    # Initialize Palmprint recognizer (HOG features)
    hog_params = {
                    'cell_size': (2, 2),      # h x w in pixels
                    'block_size': (16, 16),         # h x w in cells
                    'win_size': (64, 64),
                    'num_bins': 8
                }
    recognizer = cv.HOGDescriptor(
                            _winSize=(hog_params['win_size'][1] * hog_params['cell_size'][1],
                                    hog_params['win_size'][0] * hog_params['cell_size'][0]),
                            _blockSize=(hog_params['block_size'][1] * hog_params['cell_size'][1],
                                        hog_params['block_size'][0] * hog_params['cell_size'][0]),
                            _blockStride=(hog_params['cell_size'][1], hog_params['cell_size'][0]),
                            _cellSize=(hog_params['cell_size'][1], hog_params['cell_size'][0]),
                            _nbins=hog_params['num_bins']
                        )

    # Load database
    database = load_database(args.database_dir, detector, recognizer)

    # Initialize video stream
    device_id = 0
    cap = cv.VideoCapture(device_id)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    rot_threshold = args.rot_threshold

    # Real-time face recognition
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        # detect palms
        palms = detect_palm(detector, frame)
        # extract features
        features = extract_feature(recognizer, frame, palms)
        # match detected faces with database
        identities = []
        valids = []
        for palm, feature in zip(palms, features):
            isMatched = False
            isValid = False
            cand_identity = None
            cand_identity_dist = None

            # Check that hand rotation is within allowed range
            if np.abs(palm[-1]) < rot_threshold:
                print(np.abs(palm[-1]))
                isValid = True

                # Match palmprint with all the palmsprints in the database, and keep the closest one
                for identity, db_feature in database.items():
                    anyMatched, dist = match(recognizer, feature, db_feature)
                    print(f'identity: {identity}, dist: {dist}')
                    if anyMatched:
                        if (cand_identity_dist is None) or (dist < cand_identity_dist):
                            cand_identity = identity
                            cand_identity_dist = dist
                        isMatched = True
                
            valids.append(isValid)
            if not isMatched:
                identities.append('Unknown')
            else:
                identities.append(cand_identity)
        tm.stop()

        # Draw results on the input image
        frame = visualize(frame, palms, identities, valids, tm.getFPS())

        # Visualize results in a new Window
        cv.imshow('Face recognition system', frame)

        tm.reset()
