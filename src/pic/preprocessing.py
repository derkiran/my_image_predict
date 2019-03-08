import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle

class Preprocessing:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}
        self.split = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

    def load_data(self, filename, filetype='csv', *, name, **kwargs):
        filepath = f'{self.directory}/{filename}'

        function_name = f'read_{filetype}'
        df = getattr(pd, function_name)(filepath, **kwargs)
        self.data[name] = df
        return df.head()

    def save(self, name, filetype='csv', *, index=False, **kwargs):
        filepath = f'{self.directory}/{name}.{filetype}'
        getattr(self.data[name], f'to_{filetype}')(filepath, index=index, **kwargs)

    def cleanup(self, name, *, drop=None, drop_duplicates=False, dropna=None):
        data = self.data[name]

        if drop is not None:
            data = data.drop(columns=drop, axis=1)

        if drop_duplicates is True:
            data = data.drop_duplicates()

        if dropna is not None:
            if 'axis' not in dropna:
                dropna['axis'] = 1

            data = data.dropna(**dropna)

        self.data['clean'] = data

    def label_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        encoder = preprocessing.LabelEncoder()
        labels = pd.DataFrame()

        label_index = 0
        for column in columns:
            encoder.fit(data[column])
            label = encoder.transform(data[column])
            labels.insert(label_index, column=column, value=label)
            label_index += 1

        data = data.drop(columns, axis=1)
        data = pd.concat([data, labels], axis=1)
        self.data['clean'] = data

        return data

    def one_hot_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        categorical = pd.get_dummies(data[columns], dtype='int')
        data = pd.concat([data, categorical], axis=1, sort=False)
        self.data['clean'] = data
        return data

    def split_df(self, *, y_column, size=0.33, state=42):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']

        X = np.array(data.drop(columns=[y_column], axis=1))
        y = np.array(data[y_column].values)
        X2_train, X_test, y2_train, y_test = train_test_split(X, y, test_size= size , random_state=state)
        X_train, X_valid, y_train, y_valid = train_test_split(X2_train, y2_train, test_size=size, random_state=state)

        self.split = {'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
                      'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test}

        #return X_train, X_valid, X_test, y_train, y_valid, y_test

    def scale_data(self, *, y_scaled=False):

        if 'X_train' not in self.split:
            print('Can not find X_train. Call .split_df() first.')
            return

        scaler = preprocessing.MinMaxScaler()
        self.split['X_train'] = scaler.fit_transform(self.split['X_train'])
        self.split['X_valid'] = scaler.transform(self.split['X_valid'])
        self.split['X_test'] = scaler.transform(self.split['X_test'])

        if y_scaled is True:
            scaler = preprocessing.MinMaxScaler()
            self.split['y_train'] = scaler.fit_transform(self.split['y_train'].reshape(-1,1))
            self.split['y_valid'] = scaler.transform(self.split['y_valid'].reshape(-1,1))
            self.split['y_test'] = scaler.transform(self.split['y_test'].reshape(-1,1))
        return self.split
        picklename = 'split.'
        pickle.dump(self.split, open(picklename, 'wb'))



        #return X_train, X_valid, X_test, y_train, y_valid, y_test



    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value

