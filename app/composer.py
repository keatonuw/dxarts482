from typing import List
import random
import markovify


def create_model(dataset: str) -> markovify.Text:
    with open("./data/texts/" + dataset) as f:
        text_model = markovify.Text(f.read())
        text_model = text_model.compile()
        return text_model


def save_model(model: markovify.Text, name: str) -> None:
    with open("./data/markov-models/" + name) as f:
        json = model.to_json()
        f.write(json)


def load_models(modelpaths: List) -> List:
    models = []
    for p in modelpaths:
        with open(p) as f:
            json = f.read()
            text_model = markovify.Text.from_json(json)
            models.append(text_model)
    return models


def test_generate_article(models: List) -> str:
    # want title followed by several paragraphs. ideally in some matter of feedback?
    main_model: markovify.Text = random.choice(models)
    quote_model: markovify.Text = random.choice(models)
    title = main_model.make_short_sentence(75)
    body = ""
    while len(body) < 500:
        s = main_model.make_sentence()
        if s is str:
            body += "\n" + main_model.make_sentence()

    return f"#{title}\n{body}"
