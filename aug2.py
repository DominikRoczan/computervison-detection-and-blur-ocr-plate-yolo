import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random


# Definicja architektury Siamese Network
def create_siamese_network(input_shape):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    return models.Model(inputs=input_layer, outputs=x)


# Definicja modelu Siamese Network
def create_siamese_model(input_shape):
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    # Utworzenie dwóch identycznych "ramion"
    siamese_network = create_siamese_network(input_shape)
    output_a = siamese_network(input_a)
    output_b = siamese_network(input_b)

    # Obliczenie różnicy między reprezentacjami próbek
    diff = tf.abs(output_a - output_b)

    # Warstwa wyjściowa - sigmoidalna funkcja aktywacji
    output = layers.Dense(1, activation='sigmoid')(diff)

    return models.Model(inputs=[input_a, input_b], outputs=output)


# Definicja funkcji kosztu - kontrastowa strata
def contrastive_loss(y_true, y_pred):
    margin = 1
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))


# Ładowanie danych treningowych (np. MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizacja danych
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# Definicja par próbek treningowych
def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

# Tworzenie modelu Siamese
input_shape = x_train.shape[1:]
siamese_model = create_siamese_model(input_shape)
siamese_model.compile(optimizer='adam', loss=contrastive_loss)

# Trenowanie modelu
siamese_model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  epochs=20,
                  validation_split=0.1)

if __name__ == "__main__":
    # Ocena modelu na danych testowych
    test_loss = siamese_model.evaluate([x_test_pairs[:, 0], x_test_pairs[:, 1]], y_test,
                                       batch_size=128)
    print("Test loss:", test_loss)

    # Przykład predykcji na pojedynczej parze próbek
    sample_pair = np.expand_dims(np.array([x_test[0], x_test[1]]), axis=-1)
    prediction = siamese_model.predict([sample_pair[:, 0], sample_pair[:, 1]])
    print("Prediction for sample pair:", prediction)
