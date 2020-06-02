#!/usr/bin/env python
# coding: utf-8

import optuna
import tensorflow as tf

## 1. Define Objective

def objective_tf(trial):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    activ = trial.suggest_categorical('activation',['relu','sigmoid'])
    dropout = trial.suggest_uniform('dropout', 0, 1.)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation=activ),
      tf.keras.layers.Dropout(dropout),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5)
    loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
    print("Test Loss:", loss)
    print("Test accuracy:", accuracy)
    print("\n=================")
    return loss

## 2. Local optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective_tf, n_trials=10)


## 3. Distributed optimization
study = optuna.create_study(direction='minimize',
                            study_name='distributed-tuning_tf',
							storage='mysql+pymysql://root:root@localhost:8989/ml_expts', 
                            load_if_exists=True)

study.optimize(objective_tf, n_trials=10)
