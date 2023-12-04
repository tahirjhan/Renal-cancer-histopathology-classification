import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add,GlobalAveragePooling1D,Conv2D
from tensorflow.keras.layers import Input, Embedding, Concatenate,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

config = {}
config["num_layers"] = 12
config["hidden_dim"] = 768
config["mlp_dim"] = 3072
config["num_heads"] = 12
config["dropout_rate"] = 0.1

config["image_size"] = 256
config["patch_size"] = 32
config["num_patches"] = int((config["image_size"] // config["patch_size"])**2)
config["num_channels"] = 3
config["num_classes"] = 5

class ClassToken(Layer):
    def __init__(self,):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype="float32"),
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

def ResNet50ViT(cf):
    """ Input """
    inputs = Input((cf["image_size"], cf["image_size"], cf["num_channels"]))

    """ Pre-trained Resnet50 """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    output = resnet50.output

    """ Patch Embeddings """
    patch_embed = Conv2D(
        cf["hidden_dim"],
        kernel_size=(32, 32),  # Modified kernel_size
        padding="same"
    )(output)
    _, h, w, f = patch_embed.shape
    patch_embed = Reshape((h*w, f))(patch_embed)

    """ Position Embeddings """
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
    pos_embed = Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)

    """ Patch + Position Embeddings """
    embed = patch_embed + pos_embed

    """ Adding Class Token """
    token = ClassToken()(embed)
    x = Concatenate(axis=1)([token, embed])

    """ Transformer Encoder """
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    x = LayerNormalization()(x)
    x = x[:, 0, :]
    x = Dense(cf["num_classes"], activation="softmax")(x)

    model = Model(inputs, x)
    return model

model = ResNet50ViT(config)
model.summary()
