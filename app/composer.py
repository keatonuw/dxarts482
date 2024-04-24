from typing import List
import os
import random
import markovify


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


def test_generate_article(models: List) -> str:
    # want title followed by several paragraphs. ideally in some matter of feedback?
    main_model: markovify.Text = random.choice(models)
    quote_model: markovify.Text = random.choice(models)
    title = main_model.make_short_sentence(75)
    body = ""
    while len(body) < 500:
        s = main_model.make_sentence()
        body += f"\n{main_model.make_sentence()}"

    return f"# {title}\n{body}"


ms = []
ms.append(create_save_load("gpt-tweets"))
ms.append(create_save_load("ibos-select"))

for i in range(2):
    print(test_generate_article(ms))
