import tensorflow as tf


def identity_block(X, f, filters, training=True):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode

    Returns:
    X -- output of the identity block
    """
    x_shortcut = X
    # Retrieve Filters
    f1, f2, f3 = filters

    X = tf.keras.layers.Conv2D(filters=f1, kernel_size=1, padding='valid', strides=1)(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X, training=training)  # Default axis
    X = tf.keras.layers.ReLU()(X)

    X = tf.keras.layers.Conv2D(filters=f2, kernel_size=f, strides=(1, 1), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X, training=training)
    X = tf.keras.layers.ReLU()(X)

    X = tf.keras.layers.Conv2D(filters=f3, kernel_size=1, strides=(1, 1), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X, training=training)

    X = tf.keras.layers.Add()([X, x_shortcut])
    X = tf.keras.layers.ReLU()(X)

    return X


def convolutional_block(X, f, filters, s=2, training=True):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode

    Returns:
    X -- output of the convolutional block, tensor of shape (m, n_H, n_W, n_C)
    """

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X, training=training)
    X = tf.keras.layers.ReLU()(X)

    # Second component of main path (≈3 lines)
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X, training=training)
    X = tf.keras.layers.ReLU()(X)

    # Third component of main path (≈2 lines)
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X, training=training)

    # SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut, training=training)

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]),
    # and pass it through a RELU activation
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.ReLU()(X)

    return X


def resnet(input_shape, classes):
    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.Input(input_shape)

    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = tf.keras.layers.Conv2D(4, (3, 3), strides=(1, 1))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[8, 8, 16], s=1)
    X = identity_block(X, 3, [8, 8, 16])
    # X = identity_block(X, 3, [16, 16, 32])

    ### START CODE HERE

    # Use the instructions above in order to implement all of the Stages below
    # Make sure you don't miss adding any required parameter

    ## Stage 3 (≈4 lines)
    # `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, 3, (16, 16, 32), s=2, training=True)

    # the 3 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, (16, 16, 32), training=True)
    X = identity_block(X, 3, (16, 16, 32), training=True)

    # Stage 4 (≈6 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    # X = convolutional_block(X, 3, (256, 256, 1024), s=2, training=True)

    # # the 5 `identity_block` with correct values of `f` and `filters` for this stage
    # X = identity_block(X, 3, (256, 256, 1024), training=True)
    # X = identity_block(X, 3, (256, 256, 1024), training=True)
    # X = identity_block(X, 3, (256, 256, 1024), training=True)
    # X = identity_block(X, 3, (256, 256, 1024), training=True)
    # X = identity_block(X, 3, (256, 256, 1024), training=True)
    #
    # Stage 5 (≈3 lines)
    # add `convolutional_block` with correct values of `f`, `filters` and `s` for this stage
    X = convolutional_block(X, 3, (32, 32, 64), s=2, training=True)

    # the 2 `identity_block` with correct values of `f` and `filters` for this stage
    X = identity_block(X, 3, (32, 32, 64), training=True)
    X = identity_block(X, 3, (32, 32, 64), training=True)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D()(X)"
    X = tf.keras.layers.AveragePooling2D()(X)

    ### END CODE HERE

    # output layer
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax')(X)

    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model

def convolutional_model(input_shape):
    input_img = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(4, (3, 3), padding='same', strides=1)(input_img)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(8, (2, 2), padding='same', strides=1)(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(16, (2, 2), padding='same', strides=1)(x)

    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=4, padding='same')(x)

    x = tf.keras.layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)

    return model
