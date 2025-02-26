import tensorflow as tf 
from tensorflow.keras import layers, models

# Building an Inceptions Module. 
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):

    # 1x1 convolution. 
    conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding="same", activation="relu")(x)

    # 3x3 convolution. 
    conv_3x3 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding="same", activation="relu")(x)
    conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding="same", activation="relu")(conv_3x3)
    
    # 5x5 convolution. 
    conv_5x5 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding="same", activation="relu")(x)
    conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding="same", activation="relu")(conv_5x5)

    # MaxPooling Layer followed by 1x1 convolution.
    pool_proj = layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding="same", activation="relu")(pool_proj)

    # Concatenate filters from all branches. 
    output = layers.concatenate([
        conv_1x1, 
        conv_3x3,
        conv_5x5,
        pool_proj
    ], axis=-1)

    return output

# Building an auxillary blocks. 
def auxillary_blocks(x, num_classes):
    aux = layers.AveragePooling2D((5, 5), strides=(3, 3))(x)
    aux = layers.Conv2D(128, (1, 1), padding="same", activation="relu")(aux)
    aux = layers.Flatten()(aux)
    aux = layers.Dense(1024, activation="relu")(aux) # Fully Connected Layer. 
    aux = layers.Dropout(0.7)(aux)
    aux = layers.Dense(num_classes, activation="softmax")(aux)

    return aux

# Building an Inception Modules with two auxilliary classifiers. 
def build_googlenet(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    # Normalize the input. 
    x = layers.Rescaling(1./255)(input_layer)

    # Initial Convolution and MaxPooling. 
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Adding an LocalResponse Normalization.
    x = tf.nn.local_response_normalization()

    # Second Convolution and MaxPooling. 
    x = layers.Conv2D(_, (1, 1), padding="same", activation="relu")(x)
    x = layers.Conv2D(192, (3, 3), padding="same", activation="relu")(x)
   
    # Adding an LocalResponse Normalization.
    x = tf.nn.local_response_normalization()

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Inception Modules 1 to 4. 
    x = inception_module(x, 64, 96, 128, 16, 32, 32)  # Inception 3a
    x = inception_module(x, 128, 128, 192, 32, 96, 64)  # Inception 3b

    # MaxPooling after 3b. 
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Inception 4a.
    x = inception_module(x, 192, 96, 208, 16, 48, 64)

    #  First auxillary classifier after the 4th inception module. 
    aux1 = auxillary_blocks(x, num_classes)

    # Inception 4b
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    # Inception 4c
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    # Inception 4d
    x = inception_module(x, 112, 144, 288, 32, 64, 64)

    # Second auxillary classifier after the 7th inception module. 
    aux2 = auxillary_blocks(x, num_classes)

    # Inception 4e
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    
    # MaxPooling after 4e
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) 

    # Inception modules 8 and 9
    # Inception 5a
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    # Inception 5b
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    # Global Average Pooling and Dropout
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Final output layer (main classifier)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)

    # Define the model with inputs and three outputs (main + 2 auxiliary)
    model = models.Model(inputs=input_layer, outputs=[output_layer, aux1, aux2])
    return model

# Model parameters
input_shape = (224, 224, 3)  # ImageNet image size
num_classes = 15  # Number of classes for classification

# Build the model
googlenet = build_googlenet(input_shape, num_classes)

print(googlenet.summary())