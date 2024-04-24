# python -m flask --app server run
# flask --app app run --debug

from flask import Flask
from flask import render_template
import random

from . import composer

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

    return app
