import numpy as np
import tensorflow as tf

def lsd():
    def loss(y_true, y_pred):
        # Extract the first channel for magnitude

        # Calculate the difference per item
        diff_per_item = y_true[..., 0:1] - y_pred[..., 0:1]

        norm_per_item = tf.norm(diff_per_item, axis=[1, 2])
        #norm_per_item = tf.norm(diff_per_item)
        # Compute the mean of these norms
        mean_norm = tf.reduce_mean(norm_per_item)

        # y_true_rad = (y_true[..., 1:2] * 2.0 - 1.0) * tf.constant(np.pi)  # Scale from [0,1] to [-π, π]
        # y_pred_rad = (y_pred[..., 1:2] * 2.0 - 1.0) * tf.constant(np.pi)  # Scale from [0,1] to [-π, π]

        # # Calculate the cyclic difference
        # phase_diff = tf.math.atan2(tf.math.sin(y_true_rad - y_pred_rad), tf.math.cos(y_true_rad - y_pred_rad)) 

        # # Compute mean squared error on the phase difference
        # phase_loss = tf.reduce_mean(tf.square(phase_diff))

        return mean_norm # + (phase_loss / tf.constant(np.pi))
    return loss

def UResNet():
    def residual_encoder_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, activation="relu", strides=(1,1)):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                   kernel_initializer="he_normal", padding="same", strides=strides)(input_tensor)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU()(x)


        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                   kernel_initializer="he_normal", padding="same")(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU()(x)


        shortcut = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(1,1),
                                          strides=strides, padding="same")(input_tensor)
        if batchnorm:
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.PReLU()(x)

        return x

    def decoder_block(input_tensor, skip_tensor, n_filters, kernel_size=3, batchnorm=True, activation="relu"):
      x = tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(3, 3), strides=(2, 2), padding="same")(input_tensor)

      # Lambda layer to dynamically resize the skip_tensor
      resize_layer = tf.keras.layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))
      skip_tensor_resized = resize_layer([skip_tensor, x])

      x = tf.keras.layers.concatenate([skip_tensor_resized, x], axis=-1)
      x = residual_encoder_block(x, n_filters, kernel_size, batchnorm, activation)
      return x

    inputs = tf.keras.layers.Input(shape=(1025,87,2))
    original_phase = tf.keras.layers.Lambda(lambda x: x[..., 1:])(inputs)
    x_mag = tf.keras.layers.Lambda(lambda x: x[..., 0:])(inputs)

    x_mag = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='same')(x_mag)
    x_mag = tf.keras.layers.PReLU()(x_mag)
    x_mag = tf.keras.layers.BatchNormalization()(x_mag)

    
    # original_phase = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='same')(original_phase)
    # original_phase = tf.keras.layers.PReLU()(original_phase)
    # original_phase = tf.keras.layers.BatchNormalization()(original_phase)

    # Encoder (Downsampling using strided convolutions) - Magnitude
    ## \/ \/ \/ THIS WAS CHANGED (TURNED ON) \/ \/ \/
    #x_mag = tf.keras.layers.GaussianNoise(0.1)(x_mag) # Play with this
    c1_mag = residual_encoder_block(x_mag, 64, strides=2)
    c2_mag = residual_encoder_block(c1_mag, 128, strides=2)
    c3_mag = residual_encoder_block(c2_mag, 256, strides=2)
    c4_mag = residual_encoder_block(c3_mag, 512, strides=2)
    c5_mag = residual_encoder_block(c4_mag, 1024, strides=2)

    # Decoder (Upsampling) - Magnitude
    u5_mag = decoder_block(c5_mag, c4_mag, 1024)
    u4_mag = decoder_block(u5_mag, c3_mag, 512)
    u3_mag = decoder_block(u4_mag, c2_mag, 256)
    u2_mag = decoder_block(u3_mag, c1_mag, 128)
    u1_mag = decoder_block(u2_mag, x_mag, 64)  # Here, we're using x_mag as the skip connection

    # Output layer - Magnitude
    output_mag = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u1_mag)
    


    # # Encoder (Downsampling using strided convolutions) - Phase

    # c1_phase = residual_encoder_block(original_phase, 64, strides=2)
    # c2_phase = residual_encoder_block(c1_phase, 128, strides=2)
    # c3_phase = residual_encoder_block(c2_phase, 256, strides=2)
    # c4_phase = residual_encoder_block(c3_phase, 512, strides=2)
    # # c5_phase = residual_encoder_block(c4_phase, 1024, strides=2)

    # # Decoder (Upsampling) - Phase
    # # u5_phase = decoder_block(c5_phase, c4_phase, 1024)
    # u4_phase = decoder_block(c4_phase, c3_phase, 512)
    # u3_phase = decoder_block(u4_phase, c2_phase, 256)
    # u2_phase = decoder_block(u3_phase, c1_phase, 128)
    # u1_phase = decoder_block(u2_phase, original_phase, 64)

    # # Output layer - Phase
    # output_phase = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u1_phase)

    desired_shape = (1025, 87)

    # Cropping the magnitude output
    output_mag_cropped = tf.keras.layers.Lambda(
        lambda x: tf.image.crop_to_bounding_box(x,
                                                offset_height=0,
                                                offset_width=0,
                                                target_height=desired_shape[0],
                                                target_width=desired_shape[1])
    )(output_mag)

    # # Cropping the phase output
    # output_phase_cropped = tf.keras.layers.Lambda(
    #     lambda x: tf.image.crop_to_bounding_box(x,
    #                                             offset_height=0,
    #                                             offset_width=0,
    #                                             target_height=desired_shape[0],
    #                                             target_width=desired_shape[1])
    # )(output_phase)

    # Then combine them
    combined_output_cropped = tf.keras.layers.Concatenate(axis=-1)([output_mag_cropped, original_phase])

    model = tf.keras.models.Model(inputs=[inputs], outputs=[combined_output_cropped])
    model.compile(optimizer='adam', loss=lsd(), metrics=['accuracy',tf.keras.metrics.MeanSquaredError()])
    model.summary()
    return model

# EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Metric to monitor, usually val_loss or val_accuracy
    patience=10,           # Number of epochs with no improvement after which training will be stopped
    verbose=1,            # When set to 1, it prints a message when stopping
    restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity
)

def train_model(mp3_data, wav_data):
    ResNet = UResNet()
    print("Model Built")

    print("Starting fitting")
    # Include the EarlyStopping callback in the fit method
    ResNet.fit(
        mp3_data, 
        wav_data, 
        validation_split=0.2, 
        epochs=100,  # You can set a higher number since early stopping will be used
        batch_size=1, 
        shuffle=True, 
        callbacks=[early_stopping]  # Add the callback here
    )

    print("Finished fitting")
    return ResNet
