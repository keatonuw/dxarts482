# python -m flask --app server run
# flask --app app run --debug

from flask import Flask
from flask import send_file
from flask import request
from diffusers import StableDiffusionImg2ImgPipeline
import torch

from . import promptsynth as ps

model_path = "cw/"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, torch_dtype=torch.float16
)

# TODO: need to check what is available!!!
if torch.cuda.is_available():
    pipe.to("cuda")
elif torch.backends.mps.is_available():
    pipe.to("mps")  # or CUDA on windows/linux

synth = ps.PromptSynth(pipe)


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def main():
        return synth.generate_page()

    @app.route("/health")
    def health():
        return "I AM ALIVE!"

    @app.route("/log", methods=["POST"])
    def log():
        # TODO: log data from the client. will be some collection of positions.
        content = request.json
        if content is None:
            return "error"
        if "pos" in content:
            print(content["pos"])
            # TODO: parse content into list of (x, y, w, h) tuples
            positions = []
            for p in content["pos"]:
                positions.append((p["x"], p["y"], p["w"], p["h"]))
            synth.consume_positions(positions)

            return "ok"
        return "error"

    @app.route("/state")
    def state():
        return synth.get_state()

    @app.route("/image")
    def image():
        bts = synth.generate_image()
        return send_file(
            bts,
            mimetype="image/png",
            download_name="gen.png",
            as_attachment=True,
        )

    return app
