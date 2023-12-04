import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add,GlobalAveragePooling1D,Conv2D
from tensorflow.keras.layers import Input, Embedding, Concatenate,Reshape
from tensorflow.keras.models import Model

class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls

def mlp(x, cf):
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    return x

def transformer_encoder(x, cf):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
        num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
    )(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, cf)
    x = Add()([x, skip_2])

    return x

def ViT(input_shape=(256, 256, 3), num_classes=10):
    config = {}
    config["num_layers"] = 12
    config["hidden_dim"] = 768
    config["mlp_dim"] = 3072
    config["num_heads"] = 12
    config["dropout_rate"] = 0.1
    patch_size = 16  # Adjust the patch size based on your preference and requirements
    config["num_patches"] = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    cf = config

    """ Inputs """
    inputs = Input(input_shape)

    """ Patch Embeddings """
    patch_shape = (patch_size, patch_size)
    patch_embed = Conv2D(cf["hidden_dim"], patch_shape, strides=patch_size, padding="valid")(inputs)
    patch_embed = Reshape((cf["hidden_dim"],))(patch_embed)

    """ Position Embeddings """
    positions = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(tf.range(cf["num_patches"]))
    positions = Add()([patch_embed, positions])

    """ Transformer Encoder """
    x = positions
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    """ Classification Head """
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, x)
    return model
