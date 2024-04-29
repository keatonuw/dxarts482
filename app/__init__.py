# python -m flask --app server run
# flask --app app run --debug

from flask import Flask
from flask import render_template
from flask import send_file
import io
from diffusers import StableDiffusionPipeline
import random

from . import composer

pipe = StableDiffusionPipeline.from_pretrained("cw/")
pipe.to("mps")  # or CUDA on windows/linux

composer.init()
print("MODELS LOADED")


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def main():
        title, body = composer.gen()
        return render_template(
            "musing.html",
            title=title,
            body=body,
            style=random.choice(["dark", "light", "matrix", "blue"]),
        )

    @app.route("/image")
    def image():
        prompt = "dark, scary, grainy cctv footage l34ks of an urban city street"
        image = pipe(prompt, num_inference_steps=5, guidance_scale=7.5).images[0]
        bts = io.BytesIO()
        image.save(bts, format="jpeg")
        bts.seek(0)
        return send_file(
            bts,
            mimetype="image/jpeg",
            download_name="gen.jpg",
            as_attachment=True,
        )

    return app
