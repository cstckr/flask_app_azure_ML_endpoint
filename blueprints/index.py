from flask import Blueprint, render_template, flash
from PIL import Image
import urllib.request
import json
from extras.functions import PIL_image_to_json
from extras.forms import UploadForm
from credentials.endpoint import REST_endpoint, endpoint_token
from flask_login import login_required


index = Blueprint("index", __name__)


@index.route("/main", methods=["GET", "POST"])
@login_required
def main():
    form = UploadForm()
    if form.validate_on_submit():
        # Check and preprocess image
        try:
            img_pil = Image.open(form.file.data)
            img_mode = img_pil.mode
            img_size = img_pil.size
            if img_mode != "L":
                raise TypeError("Image must be single-channel")
            if img_size[-1] != img_size[-2]:
                raise TypeError("Image must be quadratic")
            img_pil = img_pil.resize((28, 28))
            img_json_str = PIL_image_to_json(img_pil)
        except Exception as error:
            flash(f"Please check your file. Error: {error}")
            return render_template("main.html", form=form)

        # Prediction
        dic = {"data": img_json_str}
        body = str.encode(json.dumps(dic))

        headers = {"Content-Type": "application/json",
                   "Authorization": ("Bearer " + endpoint_token)}
        req = urllib.request.Request(REST_endpoint, body, headers)
        try:
            response = urllib.request.urlopen(req)
            result = json.loads(response.read())
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " +
                  str(error.code))
        return render_template("output.html", result=result)

    elif form.is_submitted():
        flash("Only .jpg files")
    return render_template("main.html", form=form)
