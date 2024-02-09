import tensorflow as tf
import numpy as np


@tf.keras.saving.register_keras_serializable()
class DynamicResizeLayer(tf.keras.layers.Layer):
    def call(self, inputs, target_shape):
        # Resize the input to the target shape
        return tf.image.resize(inputs, size=(target_shape[1], target_shape[2]))
    def get_config(self):
        # Return an empty config dictionary since this layer has no configurable parameters
        return {}

@tf.keras.saving.register_keras_serializable()
def log_spectral_distance():
    @tf.keras.saving.register_keras_serializable()
    def loss(y_true, y_pred):
        # Extract the first channel for magnitude
        y_true_mag = y_true[..., 0:1]
        y_pred_mag = y_pred[..., 0:1]

        # Calculate the difference per item
        diff_per_item = y_true_mag - y_pred_mag

        norm_per_item = tf.norm(diff_per_item, axis=[1, 2])

        # Compute the mean of these norms
        mean_norm = tf.reduce_mean(norm_per_item)

        return mean_norm

    return loss

def scaled_tanh(x):
    return (tf.math.tanh(x) + 1) / 2

def UResNet():
    def residual_encoder_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, activation="prelu", strides=(1,1)):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                  kernel_initializer="he_normal", padding="same", strides=strides)(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation == "relu":
            x = tf.keras.layers.ReLU()(x)
        elif activation == "prelu":
            # Instantiate PReLU layer
            prelu = tf.keras.layers.PReLU()
            x = prelu(x)

        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                  kernel_initializer="he_normal", padding="same")(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation == "relu":
            x = tf.keras.layers.ReLU()(x)
        elif activation == "prelu":
            # Instantiate PReLU layer
            prelu = tf.keras.layers.PReLU()
            x = prelu(x)

        shortcut = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1),
                                          strides=strides, padding="same")(input_tensor)
        if batchnorm:
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        if activation == "relu":
            x = tf.keras.layers.ReLU()(x)
        elif activation == "prelu":
            # Instantiate PReLU layer
            prelu = tf.keras.layers.PReLU()
            x = prelu(x)

        return x

    def decoder_block(input_tensor, skip_tensor, n_filters, kernel_size=3, batchnorm=True, activation="relu"):
        x = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding="same")(input_tensor)

        # Use a Lambda layer to get the dynamic shape
        dynamic_shape = tf.keras.layers.Lambda(lambda x: tf.shape(x))(x)

        # Custom layer for dynamic resizing
        dynamic_resize_layer = DynamicResizeLayer()
        skip_tensor_resized = dynamic_resize_layer(skip_tensor, dynamic_shape)

        x = tf.keras.layers.concatenate([skip_tensor_resized, x], axis=-1)
        x = residual_encoder_block(x, n_filters, kernel_size, batchnorm)
        return x

    def process_with_lstm(tensor):
        # Extract spatial dimensions and channels
        height, width, channels = tensor.shape[1], tensor.shape[2], tensor.shape[3]

        # Reshape for LSTM: Treat width as timesteps, height * channels as features
        x = tf.keras.layers.Reshape((width, height * channels))(tensor)

        # Apply LSTM
        x = tf.keras.layers.LSTM(height * width, return_sequences=True)(x)

        # Calculate the new number of channels for reshaping
        # The total number of elements in the output shape should match input
        new_channels = 1

        # Reshape back to a 3D tensor
        x = tf.keras.layers.Reshape((height, width, new_channels))(x)

        return x


    inputs = tf.keras.layers.Input(shape=(1025,173,2))
    original_phase = tf.keras.layers.Lambda(lambda x: x[..., 1:])(inputs)
    x_mag = tf.keras.layers.Lambda(lambda x: x[..., 0:])(inputs)

    x_mag = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='same')(x_mag)
    x_mag = tf.keras.layers.PReLU()(x_mag)
    x_mag = tf.keras.layers.BatchNormalization()(x_mag)

    x_mag_lstm = process_with_lstm(x_mag)
    # Encoder (Downsampling) - Magnitude
    c1_mag = residual_encoder_block(x_mag_lstm, 64, strides=2)

    # lstm_units_c1_mag = 44631
    c1_mag_lstm = process_with_lstm(c1_mag)

    c2_mag = residual_encoder_block(c1_mag, 128, strides=2)

    # lstm_units_c2_mag = 11308
    c2_mag_lstm = process_with_lstm(c2_mag)

    c3_mag = residual_encoder_block(c2_mag, 256, strides=2)


    c3_mag_lstm = process_with_lstm(c3_mag)

    c4_mag = residual_encoder_block(c3_mag, 512, strides=2)

    c4_mag_lstm = process_with_lstm(c4_mag)

    c5_mag = residual_encoder_block(c4_mag, 1024, strides=2)

    c5_mag_lstm = process_with_lstm(c5_mag)





    # Decoder blocks: Use *_lstm outputs in skip connections
    u5_mag = decoder_block(c5_mag_lstm, c4_mag_lstm, 1024)
    u4_mag = decoder_block(u5_mag, c3_mag_lstm, 512)
    u3_mag = decoder_block(u4_mag, c2_mag_lstm, 256)
    u2_mag = decoder_block(u3_mag, c1_mag_lstm, 128)
    u1_mag = decoder_block(u2_mag, x_mag_lstm, 64)  # using x_mag directly here

    # Output layer - Magnitude
    output_mag = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u1_mag)


    desired_shape = (1025, 173)

    # Cropping the magnitude output
    output_mag_cropped = tf.keras.layers.Lambda(
        lambda x: tf.image.crop_to_bounding_box(x,
                                                offset_height=0,
                                                offset_width=0,
                                                target_height=desired_shape[0],
                                                target_width=desired_shape[1])
    )(output_mag)


    combined_output_cropped = tf.keras.layers.Concatenate(axis=-1)([output_mag_cropped, original_phase])

    model = tf.keras.models.Model(inputs=[inputs], outputs=[combined_output_cropped])
    model.compile(optimizer='adam', loss=log_spectral_distance(), metrics=['accuracy',tf.keras.metrics.MeanSquaredError()])
    model.summary()
    return model