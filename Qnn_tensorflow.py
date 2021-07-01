import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import random
import matplotlib.pyplot as plt
import numpy as np

qubit = cirq.GridQubit(0,0)
train, test = [], []
train_label, test_label = [], []

len_of_train = 5000
len_of_test = 1000
    
def prepare_data(n1, n2):
    
    for _ in range(n1):
        rdm_data = random.uniform(0,0.1) if random.random() < 0.5 else random.uniform(0.9,1)
        cir = cirq.Circuit()
        cir.append([cirq.X(qubit)**rdm_data])
        train.append(cir)
        if rdm_data < 0.5:
            train_label.append(1)
        else:
            train_label.append(-1)
    for _ in range(n2):
        rdm_data = random.uniform(0,0.1) if random.random() < 0.5 else random.uniform(0.9,1)
        cir = cirq.Circuit()
        cir.append([cirq.X(qubit)**rdm_data])
        test.append(cir)
        if rdm_data < 0.5:
            test_label.append(1)
        else:
            test_label.append(-1)
    return tfq.convert_to_tensor(train), np.array(train_label), tfq.convert_to_tensor(test), np.array(test_label)

def assemble_circuit(qubit):
    x = sympy.symbols('X_rot')
    y = sympy.symbols('Y_rot')
    z = sympy.symbols('Z_rot')
    c = cirq.Circuit()
    c.append(cirq.rx(x).on(qubit))
    c.append(cirq.ry(y).on(qubit))
    c.append(cirq.rz(z).on(qubit))
    return c

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

train, train_target, test, test_target = prepare_data(len_of_train, len_of_test)

qubit = cirq.GridQubit(0,0)
model_readout = [cirq.X(qubit)]
plt_circuit = assemble_circuit(qubit)
print(plt_circuit)

inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

output = tfq.layers.PQC(assemble_circuit(qubit), model_readout, repetitions=64, differentiator=tfq.differentiators.ParameterShift(), initializer=tf.keras.initializers.Zeros)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.keras.losses.hinge, metrics=[hinge_accuracy])

history = model.fit(train, train_target, epochs=100, batch_size=32, validation_data=(test, test_target))

print(model.trainable_weights)

plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Hinge Loss")
plt.show()


plt.plot(history.history['hinge_accuracy'], label='Training')
plt.plot(history.history['val_hinge_accuracy'], label='Validation Acc')
plt.legend()
plt.title("Training Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show() 