[![Downloads](https://pepy.tech/badge/complex)](https://pepy.tech/project/complex)
[![Downloads](https://pepy.tech/badge/complex/month)](https://pepy.tech/project/complex/month)
[![Downloads](https://pepy.tech/badge/complex/week)](https://pepy.tech/project/complex/week)

# Hyperoptim
> Hyperparameter Optimization Using Genetic Algorithm.

## Installation

OS X , Windows & Linux:

```sh
pip install hyperoptim
```
## Usage example
Use for find best hyperparameter

```python
from hyperoptim import GASearch, Hparams
import tensorflow as tf
from tensorflow import keras

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# define hyperparameter space
ht = Hparams()
hp_units = ht.Int('units', min_value=32, max_value=512, step=32)
hp_learning_rate = ht.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
hp_activation = ht.Choice('activation', values=['relu', 'sigmoid', 'tanh'])

# create the list of hyperparameter
params = [hp_units, hp_learning_rate, hp_activation]

# define model 
params = [hp_units, hp_learning_rate, hp_activation]
def model_builder(params):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    # here params[0] refer to hp_units and params[2] refer to hp_activation
    model.add(keras.layers.Dense(units=params[0], activation=params[2]))
    model.add(keras.layers.Dense(10))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params[1]),
                    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    return model

# intialize the GASearch
tuner = GASearch(model_builder=model_builder, params=params, objective='val_accuracy', weights=(1.0,), max_epochs=10, directory='my_dir', project_name='intro_to_kt')

# run the search                  
tuner.search(img_train, label_train, epochs=2, validation_split=0.2)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters()[0]

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.build(best_hps)
history = model.fit(img_train, label_train, epochs=2, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

eval_result = model.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)
```

## Development setup
For local development setup

```sh
git clone https://github.com/deepak7376/hyperoptim
cd hyperoptim
pip install -r requirements.txt
```

## Meta
Deepak Yadav

Distributed under the MIT license. See ``LICENSE`` for more information.
[https://github.com/deepak7376/hypertune/blob/master/LICENSE](https://github.com/deepak7376)

## References
None