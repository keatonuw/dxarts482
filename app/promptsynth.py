from flask import render_template
from PIL import Image, ImageDraw
import numpy as np
from typing import Text
import random
import io

from . import composer

states = [
    "web",
    "qr",
    "code",
    "local",
    "eigen",
    "ai",
]


class PromptSynth:
    def __init__(self, pipe):
        composer.init()
        self.pipe = pipe
        self.entities = []
        self.state = states[0]

    # Generates a page of text content based on the current state
    def generate_page(self) -> Text:
        title, body = composer.gen()
        return render_template(
            "musing.html",
            title=title,
            body=body,
            style=random.choice(["dark", "light", "matrix", "blue"]),
        )

    # Generates an image response based on current state
    def generate_image(self) -> io.BytesIO:
        prompt = self.__get_text_prompt()
        ref = self.__get_image_prompt()
        image = self.pipe(
            prompt=prompt,
            image=ref,
            strength=0.5,
            num_inference_steps=50,
            guidance_scale=7.5,
            negative_prompt="unclear, wall, plain, empty",
        ).images[0]
        bts = io.BytesIO()
        image.save(bts, format="jpeg")
        # ref.save(bts, format="jpeg")
        bts.seek(0)
        return bts

    # Consume positions (a list of (x, y, w, h) rectangles) to inform state
    def consume_positions(self, positions):
        # forget the past
        prev_amt = len(self.entities)
        self.entities = [
            (x / 1280 * 256, y / 720 * 256, w / 1280 * 256, h / 720 * 256)
            for x, y, w, h in positions
        ]
        cur_amt = len(self.entities)
        delta = cur_amt - prev_amt

        # determine the future
        if self.state == "ai":
            if delta >= 0 and cur_amt > 1:
                self.state = "ai"
            else:
                self.state = random.choice(["eigen", "local"])
        elif self.state == "eigen":
            if delta >= 0 and cur_amt > 1:
                self.state = "eigen"
            else:
                self.state = random.choice(["local", "web"])
        elif self.state == "local":
            if delta >= 0:
                self.state = "ai"
            else:
                self.state = random.choice(["web", "local"])
        elif self.state == "web":
            if delta >= 0:
                self.state = random.choice(["web", "qr", "code", "local"])
            else:
                self.state = random.choice(["web", "qr", "code"])
        else:  # code or qr
            self.state = random.choice(["web", "qr", "code"])

    # Gets the current system state, as a string for TD client to interpret
    def get_state(self) -> Text:
        # current state should be a function of entities
        # should roughly inverse-mirror human activity
        # (the machine is shy, yet it craves attention)
        # States:
        # with >= 1 rectangle(s):
        # -> ai: display generated images (need > 1 rectangles)
        # -> eigen: display eigenfaces (need a face, 1 small rectange)

        # as a transition from rectangles to none
        # -> local: display locally synthesized visuals, spliced with behind-the-scenes & plundered footage

        # with no rectangles, randomly select from:
        # -> code: display stripped code snippets from the code base
        # -> qr: display qr code to access website
        # -> web: display generate pages on the website
        return self.state

    def __get_text_prompt(self):
        return (
            "digital hacked city security camera overlooking an empty space. dark and glitchy cctv street footage l34ks at night. black and white grainy and lossy footage. "
            + composer.prompt()
        )

    def __get_image_prompt(self):
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 4), dtype=np.dtype("uint8"))
        )
        draw = ImageDraw.Draw(img, "RGBA")
        for x, y, w, h in self.entities:
            draw.rectangle([x, y, x + w, y + h], fill="#aaaaaacc", outline="black")
        bg = Image.new("RGB", img.size, "gray")
        bg.paste(img, mask=img.split()[3])
        return bg
