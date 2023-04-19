def train_spacy(train_data, iterations):
    spacy.require_gpu()
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label("TEST")

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize()
        for itn in range(iterations):
            print(f"Starting iteration {str(itn)}")
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = spacy.training.Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, losses=losses)
            print(losses)
    return nlp

nlp = train_spacy(train_data, 5)

nlp.to_disk("trained_ner_model")
