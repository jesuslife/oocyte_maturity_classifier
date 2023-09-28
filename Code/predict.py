import cv2
import torch
from torchvision import models
import torchvision.transforms as tvt
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys


def main():
    # Gets the user parameters
    parser = argparse.ArgumentParser(description='script to discriminate the presence or absence of the polar body')
    parser.add_argument('--input', help='path/to/input <image file>', type=str, required=True)
    parser.add_argument('--output', help='path/to/output <text file>', type=str, required=True)
    args = parser.parse_args()

    img_name = args.input
    txt_name = args.output
    base_name = os.path.basename(img_name).split('.')[0]

    out_dir = os.path.dirname(txt_name)
    if out_dir == '':
        out_dir = './'
        txt_name = os.path.join(out_dir, txt_name)

    # Sets the model weights
    det_weights_path = os.path.join(WEIGHT_DIR, DET_WEIGHT_FILE)
    clf_weights_path = os.path.join(WEIGHT_DIR, CLF_WEIGHT_FILE)

    # Checks the different src/dst paths
    if not os.path.exists(det_weights_path):
        sys.exit('Error: path to "%s" does not exist' % det_weights_path)

    if not os.path.exists(clf_weights_path):
        sys.exit('Error: path to "%s" does not exist' % clf_weights_path)

    if not os.path.exists(img_name):
        sys.exit('Error: path to "%s" does not exist' % img_name)

    if not os.path.exists(os.path.dirname(txt_name)):
        sys.exit('Error: path to "%s" does not exist' % txt_name)

    try:
        # Sets a device (CPU or GPU)
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Creates the detection model (YOLOv5)
        print('Detecting object...')
        det_model = torch.hub.load('.',
                                   'custom',
                                   path=det_weights_path,
                                   force_reload=True,
                                   source='local')

        # Puts the detection model in a device (CPU or GPU)
        det_model = det_model.to(device)

        # Initializes the detection model
        det_model.compute_iou = 0.6  # IoU threshold
        det_model.conf = 0.6  # Confidence threshold
        det_model.max_det = 1  # Max number of detections

        # Reads the input image
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        '''
        # Converts the color image to grayscale format
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        
        # Applies the CLAHE algorithm for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = np.stack((img,) * 3, axis=-1)
        '''

        # Sets the detector to evaluation mode
        det_model.eval()

        # Performs object detection
        results = det_model(img)
        detections = results.xyxy[0].detach().cpu().numpy()

        # Checks if there is oocyte detections:
        if len(detections) > 0:
            box = detections[0][:4]
            x, y, w, h = round(box[0]), round(box[1]), round(box[2] - box[0]), round(box[3] - box[1])
        else:
            sys.exit('No oocyte detection')
        print('Done\n')

        # Creates the classification model
        clf_model = models.resnet101()

        # Adds a new final layer
        nr_filters = clf_model.fc.in_features  # Number of input features of last layer
        clf_model.fc = nn.Linear(nr_filters, N_OUTPUT_NEURONS)

        # Loads the trained model
        clf_model.load_state_dict(torch.load(clf_weights_path))

        # Puts the classification model in a device (CPU or GPU)
        clf_model = clf_model.to(device)

        # Sets the classifier to evaluation mode
        clf_model.eval()

        # Crops the image
        crop_img = img[y:y + h, x:x + w]

        # Composes the image transformations
        transform_img = tvt.Compose([tvt.ToPILImage(),
                                     tvt.Resize((IMG_HEIGHT, IMG_WIDTH), tvt.InterpolationMode.BILINEAR),
                                     tvt.Grayscale(num_output_channels=3),
                                     tvt.ToTensor()])

        # Applies the transformations to the cropped image
        crop_img = transform_img(crop_img)

        # Puts the transformed image in a device
        crop_img = torch.autograd.Variable(crop_img, requires_grad=False).to(device).unsqueeze(0)

        # Performs classification
        print('Discriminating the presence/absence of the polar body ...')
        with torch.no_grad():
            prd = clf_model(crop_img.float())

        # Gets the confidence and label tensors
        confidence = torch.sigmoid(prd)
        label = (confidence > 0.5).type(torch.uint8)

        # Puts the confidence and label tensors in numpy arrays and extracts the scalar values
        confidence = confidence.data.cpu().detach().numpy().squeeze(0)[0]
        label = label.data.cpu().detach().numpy().squeeze(0)[0]

        if label == 1:
            msg = 'the polar body is present'
        else:
            msg = 'the polar body is not present'

        # Saves the predicted label to a text file
        print('Confidence=%.2f\nLabel=%d\nResult: %s' % (confidence, label, msg))
        output = open(txt_name, "w")
        output.write(str(label))
        output.close()
        print('Done')

        # Saves the cropped image with a summary of the prediction
        fig, ax = plt.subplots()
        ax.imshow(img[y:y + h, x:x + w])
        ax.axis('off')
        ax.set_title('Confidence=%.2f Label=%d \nResult: %s' % (confidence, label, msg))
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, base_name + '_result.jpg'))
        plt.close('all')

    except Exception as e:
        print('Exception: %s' % str(e))


if __name__ == '__main__':
    # Main parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    N_OUTPUT_NEURONS = 1
    WEIGHT_DIR = '../weights'
    DET_WEIGHT_FILE = 'oocyte_det.pt'
    CLF_WEIGHT_FILE = 'polar_body_clf.pt'

    # Call to main function
    main()
