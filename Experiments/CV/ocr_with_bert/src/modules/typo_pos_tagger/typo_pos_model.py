import joblib

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from typo_pos_dataset import get_dataset


SCALER_PATH = 'data/scaler.joblib.pkl'
VOCAB_PATH = 'data/vocab.json'

def scale_data(X_train, X_val):
    scaler = MinMaxScaler()
    scaler.fit(pd.concat([X_train, X_val], axis=0))
    joblib.dump(scaler, SCALER_PATH)
    
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)   
    return X_train, X_val


def build_model(input_dim,
                hidden_neurons,
                output_dim):
    """
    Construct, compile and return a Keras model which will be used to fit/predict
    """
    model = Sequential([
        Dense(hidden_neurons, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_clf(X_train, X_val, y_train, y_val, n_epochs):
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    
    model_params = {
        'build_fn': build_model,
        'input_dim': X_train.shape[1],
        'hidden_neurons': 512,
        'output_dim': y_train.shape[1],
        'epochs': n_epochs,
        'batch_size': 256,
        'verbose': 1,
        'validation_data': (X_val, y_val),
        'shuffle': True,
        'callbacks': [EarlyStopping(monitor='val_accuracy', patience=10)]
        }
    clf = KerasClassifier(**model_params)
    return clf


def keras_clf(X_train, X_val, y_train, y_val, n_epochs):
    X_train, X_val = scale_data(X_train, X_val)
    
    clf = build_clf(X_train, X_val, y_train, y_val, n_epochs)
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    print(f'nn classifier score: {score}')
    
    clf.model.save('models/nn_clf.h5')
    return clf


def sklearn_clf(X_train, X_val, y_train, y_val, model='dt'):
    X_train, X_val = scale_data(X_train, X_val)
    
    if model == 'dt':  
        clf = DecisionTreeClassifier(criterion='entropy')
    elif model == 'rf':
        clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=3)
    elif model == 'mlp':
        clf = MLPClassifier(max_iter=100)
    clf.fit(X_train, y_train) 
    score = clf.score(X_val, y_val)
    print(f'{model} classifier score: {score}')
    _ = joblib.dump(clf, f'models/{model}_clf.joblib.pkl', compress=9)
    return clf


X_train, X_val, y_train, y_val = get_dataset('../../data/pos_tagger/amazon_corpus_5000.json',
                                             vocab_path=VOCAB_PATH,
                                             train_split=0.8,
                                             ready=True)

X_train.drop(['is_first', 'is_last'], axis=1, inplace=True)
X_val.drop(['is_first', 'is_last'], axis=1, inplace=True)


for model in ['dt', 'rf', 'mlp']:
    clf = sklearn_clf(X_train, X_val, y_train, y_val, model=model)
#clf_nn = keras_clf(X_train, X_val, y_train, y_val, 1)


