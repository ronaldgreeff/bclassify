#*-*encoding: utf-8*-*
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


if __name__ == '__main__':

    class learner():

        def __init__(self, input_data, selection=None):

            self.select_data(input_data, selection)


        def select_data(self, input_data, selection):

            input_data = pd.read_csv(input_data)
            self.input_data = input_data[input_data['site'].isin(selection)] if selection else input_data


        def standardise_binary_features(self):
            """ Not in use """

            for initial in ('S', 'F', 'Q', 'D'):

                l = []

                for column in self.input_data.columns:
                    if column.startswith(initial):
                        l.append(column)

                self.input_data[l] = StandardScaler().fit_transform(self.input_data[l])


        def standardise_continuous_features(self):

            for continous_feature in ('height', 'width', 'links', 'texts', 'images'):

                self.input_data[continous_feature] = StandardScaler().fit_transform(self.input_data[[continous_feature]])


        def PCA(self):

            pca_features = PCA(n_components=2)

            data = self.input_data.loc[:, 'height':]

            principle_components = pca_features.fit_transform(data)

            self.df_principle_components = pd.DataFrame(data=principle_components,
                columns=['pca1', 'pca2'])


        def plot(self):
            pass



    lobj = learner(input_data='processed_urls.csv', selection=['amazon'])
    lobj.standardise_continuous_features()
    lobj.PCA()