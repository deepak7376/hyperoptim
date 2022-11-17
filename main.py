from src.hypertune import GeneticAlgorithmSearch as GA
from src.hypertune import Hparams
import tensorflow as tf
from tensorflow import keras

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# define search parameters
ht = Hparams()
hp_units = ht.Int('units', min_value=32, max_value=512, step=32)
hp_learning_rate = ht.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
hp_activation = ht.Choice('activation', values=['relu', 'sigmoid', 'tanh'])

# define model 
params = [hp_units, hp_learning_rate, hp_activation]
def model_builder(params):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(units=params[0], activation=params[2]))
    model.add(keras.layers.Dense(10))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params[1]),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    return model

tuner = GA(model_builder,
            objective='val_accuracy',
            max_epochs=10,
            directory='my_dir',
            project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
                     
tuner.search(img_train, label_train, params, epochs=2, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.build(best_hps)
history = model.fit(img_train, label_train, epochs=5, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

eval_result = model.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)