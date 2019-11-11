#!/usr/bin/env python
# coding: utf-8

import sys

from flask import Flask, render_template, Response
from camera import Camera  # emulated camera


app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page which makes use of /mjpeg."""
    return render_template('index.html')

@app.route('/mjpeg')
def mjpeg():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Camera(**get_parameters()).image_request(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    direct_passthrough=True)

@app.route('/jpeg')
def jpeg():
    return Response(Camera(**get_parameters()).image_classification_render(),
                    mimetype='image/jpeg',
                    direct_passthrough=True)

def get_parameters():
    p = dict()
    nb = len(sys.argv)
    if nb >= 2: p['width']   = max(50, int(sys.argv[2]))
    if nb >= 3: p['height']  = max(50, int(sys.argv[3]))  
    return p


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False, threaded=True)
