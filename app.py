from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
import cv2
from flask import Flask, request, render_template
import imutils
import io
import numpy as np
import os
import pickle
from PIL import Image


app = Flask(__name__, template_folder='.')

# load models
detector = cv2.dnn.readNetFromCaffe(
    'models/face_detection_model/deploy.prototxt',
    'models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel',
)
embedder = cv2.dnn.readNetFromTorch('models/openface_nn4.small2.v1.t7')
recognizer = pickle.loads(open('models/recognizer.pickle', 'rb').read())
le = pickle.loads(open('models/le.pickle', 'rb').read())


def recognize(image, conf=0.5):
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > conf:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]

            for class_, pred_ in zip(le.classes_, preds):
                print(class_, pred_)

            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return {'preds': preds, 'classes': le.classes_, 'image': image}

@app.route('/')
def hello():
    return render_template('html/upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file'

        # process data
        infile = request.files['file']
        im = Image.open(io.BytesIO(infile.stream.read()))
    
        H, W = im.size[0], im.size[1]
        X = np.array(
            im.getdata()).reshape(
            W, H, 3).astype(np.uint8)
        
        print(X.shape)
        retval = recognize(X[:, :, ::-1])

        X = retval['image']
        W = X.shape[0]
        H = X.shape[1]

        img = np.empty((W, H), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((W, H, 4))

        view[:, :, :3] = X[::-1, :, ::-1]
        view[:, :, 3] = 255

        p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
        p.x_range.range_padding = p.y_range.range_padding = 0

        # must give a vector of images
        p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)

        # grab the static resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        # render template
        script, div = components(p)
        html = render_template(
            'html/bokeh.html',
            plot_script=script,
            plot_div=div,
            js_resources=js_resources,
            css_resources=css_resources,
        )
        return html
            
    else: 
        return "Upload"

if __name__ == '__main__':
    app.run()
