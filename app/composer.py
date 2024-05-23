from typing import Dict, List
import os
import random
import markovify
import re

filler_titles = [
    "",
]

filler_content = [
    "",
]

replacement_targets = [
    "ChatGPT",
    "GPT",
    "Elon",
    "Musk",
    "Bitcoin",
    "Altman",
    "OpenAI",
    "Crypto",
    "Etherium",
]

replace_values = [
    "GOD",
    "INTERNET",
    "NETWORK",
    "CONNECTION",
    "SPIRITUALITY",
    "SHE",
]


def create_save_load(dataset: str) -> markovify.Text:
    # does the model already exist? load it
    if os.path.exists(f"./data/markov-models/{dataset}.json"):
        return load_model(dataset)
    # does it not exist? make it then save and load it
    if os.path.exists(f"./data/texts/{dataset}.txt"):
        model = create_model(dataset)
        save_model(model, dataset)
        return model
    raise Exception("Invalid dataset")


def create_model(dataset: str) -> markovify.Text:
    with open(f"./data/texts/{dataset}.txt") as f:
        text_model = markovify.Text(f.read())
        text_model = text_model.compile()
        return text_model


def save_model(model: markovify.Text, name: str) -> None:
    with open(f"./data/markov-models/{name}.json", "w") as f:
        json = model.to_json()
        f.write(json)


def load_model(model: str) -> markovify.Text:
    with open(f"./data/markov-models/{model}.json") as f:
        json = f.read()
        text_model = markovify.Text.from_json(json)
        return text_model


def load_models(modelpaths: List) -> List:
    models = []
    for p in modelpaths:
        models.append(load_model(p))
    return models


def censor(s):
    for t in replacement_targets:
        s = re.sub(t, random.choice(replace_values), s, flags=re.I)
    return s


def test_generate_article(models: List):
    # want title followed by several paragraphs. ideally in some matter of feedback?
    main_model: markovify.Text = random.choice(models)
    quote_model: markovify.Text = random.choice(models)
    title = main_model.make_short_sentence(75)
    if title is None:
        title = random.choice(filler_titles)
    title = censor(title)
    body = []
    prev_mode = "quote"
    while len(body) < 20:
        m = main_model
        if prev_mode == "quote":
            prev_mode = "main"
        else:
            prev_mode = random.choice(["main", "quote", "scroll"])
            if prev_mode == "main":
                m = main_model
            else:
                m = quote_model
        s = m.make_sentence()
        if s is None:
            s = random.choice(filler_content)
        s = censor(s)
        body.append((prev_mode, f"\n{s}"))

    return (title, body)


markov_models: Dict[str, markovify.Text] = {}


def init():
    markov_models["gpt"] = create_save_load("gpt-tweets")
    markov_models["ibos"] = create_save_load("ibos-select")
    markov_models["t4t"] = create_save_load("dysphoria")
    markov_models["tulpa"] = create_save_load("tulpa")


def gen():
    return test_generate_article([markov_models[k] for k in markov_models])


def gen_with(models):
    return test_generate_article([markov_models[k] for k in models if k in models])
