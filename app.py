from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
#from bokeh.util.string import encode_utf8
from flask import Flask, request, render_template
import io
import numpy as np
from PIL import Image
from IPython import embed
import cv2
from recognize import recognize
app = Flask(__name__, template_folder='.')


@app.route('/')
def hello():
    return render_template('upload.html')

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
        args = {
            'image': X[:, :, ::-1],
            'detector': 'face_detection_model',
            'embedding_model': 'openface_nn4.small2.v1.t7',
            'recognizer': 'output/recognizer.pickle',
            'le': 'output/le.pickle', 'confidence': 0.5
        }
        retval = recognize(args)

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
            'bokeh.html',
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
