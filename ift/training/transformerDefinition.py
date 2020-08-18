import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, numHeads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.numHeads = numHeads
        if embed_dim % numHeads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {numHeads}"
            )
        self.projection_dim = embed_dim // numHeads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.numHeads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, numHeads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, numHeads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, numHeads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, numHeads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            "embedDim": self.embed_dim,
            "numHeads": self.numHeads,
            "projection_dim": self.projection_dim,
            "query_dense": self.query_dense,
            "key_dense": self.key_dense,
            "value_dense": self.value_dense,
            "combine_heads": self.combine_heads
            })
        return config
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, numHeads, numHidden, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, numHeads)
        self.ffn = keras.Sequential(
            [layers.Dense(numHidden, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def rootMeanSquare(data):
        return tf.math.l2_normalize(data, axis = 1, epsilon = 1e-6)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
            })
        return config

class TransformerBlockRMS(layers.Layer):
    def __init__(self, embed_dim, numHeads, numHidden, rate=0.1):
        super(TransformerBlockRMS, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, numHeads)
        self.ffn = keras.Sequential(
            [layers.Dense(numHidden, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def rootMeanSquare(data):
        return tf.math.l2_normalize(data, axis = 1, epsilon = 1e-6)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        #out1 = self.layernorm1(inputs + attn_output)
        out1 = rootMeanSquare(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        #return self.layernorm2(out1 + ffn_output)
        return rootMeanSquare(out1 + ffn_output)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
            })
        return config
class TransformerBlockNoNorm(layers.Layer):
    def __init__(self, embed_dim, numHeads, numHidden, rate=0.1):
        super(TransformerBlockNoNorm, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, numHeads)
        self.ffn = keras.Sequential(
            [layers.Dense(numHidden, activation="relu"), layers.Dense(embed_dim),]
        )
        #self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def rootMeanSquare(data):
        return tf.math.l2_normalize(data, axis = 1, epsilon = 1e-6)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        #out1 = self.layernorm1(inputs + attn_output)
        out1 = attn_output
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        #return self.layernorm2(out1 + ffn_output)
        return out1 + ffn_output
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
            })
        return config
    

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def transformerNetworkTest(trackShape, numHidden, numHeads):
    inputFeatures = layers.Input(trackShape)
    inputFeaturesMasked = layers.Masking(mask_value = -999, name = "maskFeatures")(inputFeatures)

    transformerBlock = TransformerBlock(trackShape[1], numHeads, numHidden)(inputFeaturesMasked)
    flattened = layers.GlobalAveragePooling1D()(transformerBlock)
    dense = layers.Dense(numHidden, activation = "relu")(flattened)
    out = layers.Dense(1, activation = "sigmoid")(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def transformerNetworkConv(trackShape, numHidden, numHeads):
    inputFeatures = layers.Input(trackShape)
    inputFeaturesMasked = layers.Masking(mask_value = -999, name = "maskFeatures")(inputFeatures)
    processedFeatures = layers.TimeDistributed(layers.Dense(numHidden, activation = 'relu'), name = 'tdDense')(inputFeaturesMasked)
    layers.Conv1D(4,2, activation = "relu", input_shape = inputFeaturesMasked.shape[1:])(processedFeatures)
    transformerBlock = TransformerBlock(numHidden, numHeads, numHidden)(inputFeaturesMasked)
    flattened = layers.GlobalAveragePooling1D()(transformerBlock)
    dense = layers.Dense(numHidden, activation = "relu")(flattened)
    out = layers.Dense(1, activation = "sigmoid")(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def transformerNetworkDeep(trackShape, numHidden, numHeads):
    inputFeatures = layers.Input(trackShape)
    inputFeaturesMasked = layers.Masking(mask_value = -999, name = "maskFeatures")(inputFeatures)
    processedFeatures = layers.TimeDistributed(layers.Dense(18, activation = 'relu'), name = 'tdDense')(inputFeaturesMasked)
    transformerBlock = TransformerBlock(trackShape[1], numHeads, numHidden)(processedFeatures)
    flattened = layers.GlobalAveragePooling1D()(transformerBlock)
    dense = layers.Dense(numHidden, activation = "relu")(flattened)
    out = layers.Dense(1, activation = "sigmoid")(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def transformerNetworkDeepRMS(trackShape, numHidden, numHeads):
    inputFeatures = layers.Input(trackShape)
    inputFeaturesMasked = layers.Masking(mask_value = -999, name = "maskFeatures")(inputFeatures)
    processedFeatures = layers.TimeDistributed(layers.Dense(18, activation = 'relu'), name = 'tdDense')(inputFeaturesMasked)
    transformerBlock = TransformerBlockRMS(trackShape[1], numHeads, numHidden)(processedFeatures)
    flattened = layers.GlobalAveragePooling1D()(transformerBlock)
    dense = layers.Dense(numHidden, activation = "relu")(flattened)
    out = layers.Dense(1, activation = "sigmoid")(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def transformerNetworkDeepNoNorm(trackShape, numHidden, numHeads):
    inputFeatures = layers.Input(trackShape)
    inputFeaturesMasked = layers.Masking(mask_value = -999, name = "maskFeatures")(inputFeatures)
    processedFeatures = layers.TimeDistributed(layers.Dense(18, activation = 'relu'), name = 'tdDense')(inputFeaturesMasked)
    transformerBlock = TransformerBlockNoNorm(trackShape[1], numHeads, numHidden)(processedFeatures)
    flattened = layers.GlobalAveragePooling1D()(transformerBlock)
    dense = layers.Dense(numHidden, activation = "relu")(flattened)
    out = layers.Dense(1, activation = "sigmoid")(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def transformerNetworkDeepFlatten(trackShape, numHidden, numHeads):
    inputFeatures = layers.Input(trackShape)
    inputFeaturesMasked = layers.Masking(mask_value = -999, name = "maskFeatures")(inputFeatures)
    processedFeatures = layers.TimeDistributed(layers.Dense(18, activation = 'relu'), name = 'tdDense')(inputFeaturesMasked)
    transformerBlock = TransformerBlock(trackShape[1], numHeads, numHidden)(processedFeatures)
    flattened = layers.Flatten()(transformerBlock)
    dense = layers.Dense(numHidden, activation = "relu")(flattened)
    out = layers.Dense(1, activation = "sigmoid")(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)

def transformerNetworkDeepMaxPooling(trackShape, numHidden, numHeads):
    inputFeatures = layers.Input(trackShape)
    inputFeaturesMasked = layers.Masking(mask_value = -999, name = "maskFeatures")(inputFeatures)
    processedFeatures = layers.TimeDistributed(layers.Dense(18, activation = 'relu'), name = 'tdDense')(inputFeaturesMasked)
    transformerBlock = TransformerBlock(trackShape[1], numHeads, numHidden)(processedFeatures)
    flattened = layers.GlobalMaxPooling1D()(transformerBlock)
    dense = layers.Dense(numHidden, activation = "relu")(flattened)
    out = layers.Dense(1, activation = "sigmoid")(dense)

    return keras.models.Model(inputs = inputFeatures, outputs = out)
    
if __name__ == '__main__':

    # model = catNetwork((100, 18,), 4)
    # model = tagNetworkEmbed((100, 18), 3)
    model = transformerNetworkTest((100, 18))
    model.summary()
