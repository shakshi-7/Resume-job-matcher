import tensorflow as tf

def cosine_similarity(vec1, vec2):
    return 1 - tf.keras.losses.cosine_similarity(vec1, vec2).numpy()
