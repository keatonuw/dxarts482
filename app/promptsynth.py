from flask import render_template
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
        prompt = self.__get_prompt()
        # TODO: make this use image to image!
        image = self.pipe(prompt, num_inference_steps=15, guidance_scale=7.5).images[0]
        bts = io.BytesIO()
        image.save(bts, format="jpeg")
        bts.seek(0)
        return bts

    # Consume positions (a list of (x, y, w, h) rectangles) to inform state
    def consume_positions(self, positions):
        # forget the past
        self.entities = positions
        # determine the future
        idx = 0
        if len(self.entities) < 1:
            # no objects
            idx = random.randint(0, 3)
        elif len(self.entities) <= 1:
            # few objects
            idx = 3
        else:
            # many objects
            idx = random.randint(4, len(states))
        self.state = states[idx]

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

    def __get_prompt(self):
        return ""
