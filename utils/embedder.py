import tensorflow_hub as hub

embed_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_embedding(text):
    return embed_model([text])[0]
