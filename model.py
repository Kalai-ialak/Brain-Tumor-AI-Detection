import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(128, 128, 1)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    b1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    u1 = layers.UpSampling2D((2, 2))(b1)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    u2 = layers.UpSampling2D((2, 2))(c3)
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c4)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = unet_model()
model.summary()
print("\nU-Net Model Created Successfully!")