#*-*encoding: utf-8*-*
from peewee import *
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer, OneHotEncoder
import pandas as pd
import numpy as np
import json
import re
from urllib.parse import urlparse, parse_qs
import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('punkt')


re_toke = re.compile('\W+')

database = SqliteDatabase('raw_extracts.db')

class BaseModel(Model):
    class Meta:
        database = database

class Extract(BaseModel):
    url = CharField(null=True, unique=True)
    site = CharField(null=True)
    screenshot = CharField(null=True)
    extract = TextField(null=True)


def convert_extract_to_csv(filename, limited=True):

    l = []

    for select in Extract.select().where(Extract.extract.is_null(False)):

        site = select.site

        extract = json.loads(select.extract)

        d = {
            'site': site,
            'url': extract['url'],
            'images': len(extract['images']),
            'texts': len(extract['texts']),
            'links': len(extract['links']),
        }

        if limited:

            d.update(
                {k: ''.join([i for i in v if not i.isalpha()]) for k, v in extract['body']['computed'].items() if k in ('height', 'width')})

            meta_tags = extract['meta_tags']

            descriptions = set()
            titles = set()
            types = set()

            for title_string in [t for t in extract['titles'] if t]:
                for word in re.split(re_toke, title_string):
                    word = word.lower()
                    if word and (len(word)>1) and word.isalpha() and word != site:
                        titles.add(word)

            for k in meta_tags:

                # Abstract this. Look into using regex for description and title

                if k in ('keywords', 'description', 'og:description', 'twitter:description'):
                    descript_string = meta_tags[k]

                    for word in re.split(re_toke, descript_string):
                        word = word.lower()
                        if word and (len(word)>1) and word.isalpha() and word != site:
                            descriptions.add(word)

                elif k in ('og:title', 'twitter:title'):
                    title_string = meta_tags[k]

                    for word in re.split(re_toke, title_string):
                        word = word.lower()
                        if word and (len(word)>1) and word.isalpha() and word != site:
                            titles.add(word)

            if descriptions:
                d['descriptions'] = ','.join(descriptions)

            if titles:
                d['titles'] = ','.join(titles)


        else:

            d.update(
                {k:v for k, v in extract['meta_tags'].items()})

            d.update(
                {k:v for k, v in extract['body']['computed'].items()})

        l.append(d)


    df = pd.DataFrame(l)

    df.to_csv(filename, index=None, header=True)


if __name__ == '__main__':

    class feature_extractor():

        def __init__(self):
            self.classification_data = pd.read_csv("page_classification_data.csv")

            self.lb_site = LabelBinarizer()
            self.lb_frag = LabelBinarizer()
            self.mlb_query = MultiLabelBinarizer()
            self.mlb_descriptions = MultiLabelBinarizer()
            # self.oe.path = OrdinalEncoder()

            self.get_stopwords()
            self.get_page_descriptors()


        def get_stopwords(self):

            alphabet = list(map(chr, range(97, 123)))
            numbers = list(map(str, range(10)))
            additional_stopwords = ['john', 'lewis', 'partners' 'marks', 'spencer', 'en', 'gb']

            self.stop_words = alphabet + numbers + additional_stopwords


        def get_page_descriptors(self):

            def reduce_descriptors():

                df = pd.DataFrame({
                    'site': self.classification_data['site'],
                    'titles': self.classification_data['titles'],
                    'descriptions': self.classification_data['descriptions']
                    })

                for subset in ('titles', 'descriptions'):
                    df = df[df[subset].notnull()]

                    df[subset] = df[subset].apply(lambda x: [item for item in x.split(',') if item not in self.stop_words])

                    # TODO
                    # Tokenize each desciptor word

                    for site in df.site.unique():
                        description_lists = [i for i in df.loc[df['site'] == site][subset]]

                        s = set()
                        for description_list in description_lists:
                            s = s.union(set(description_list))

                        for si in s:
                            print(si)

                    break
                    # p = set()
                    # for l in ll:
                    #   p = p.union(set(l))

                    # l = [[1,2,3], [3,4,5]]
                    # search = [1,3,4]
                    # for s in search:
                    #     print(all([s in x for x in l]))

            # df = reduce_descriptors() # Doesn't seem like this is useful yet


            df = pd.DataFrame({
                'site': self.classification_data['site'],
                'descriptions': self.classification_data['descriptions'] + ',' + self.classification_data['titles']
            }).fillna('empty')

            df['descriptions'] = df['descriptions'].apply(
                lambda x: [item for item in x.split(',') if item not in self.stop_words])

            self.df_descriptors = df


        def extract_url_features(self, url):

            u = urlparse(url)
            string = ''.join(u[2:]) # exclude scheme & netloc

            cu = []
            cl = []
            n = []
            s = []

            for i in string:
                if i.isalpha():
                    if i.isupper():
                        cu.append(i)
                    else:
                        cl.append(i)
                elif i.isnumeric():
                    n.append(i)
                else:
                    s.append(i)

            # TODO:
            # Path contains useful classifying info, e.g. /women/a-to-z-of-brands/adidas/cat/
            # ordinal left to right

            queries = [i if i else '' for i in parse_qs(u.query).keys()]

            d = {
                'netloc': u[1],
                'length': len(string),
                'counts': {
                    'lower': len(cu),
                    'upper': len(cl),
                    'numeric': len(n),
                    'symbols': len(s)
                },
                'segments': len(u.path.split('/')),
                'queries': queries,
                'fragments': u[5],
            }

            return d


        def encode_url_features(self):

            df = pd.DataFrame({'url_features': self.classification_data['url'].apply(lambda x: self.extract_url_features(x))})

            binary_sites = ('S', self.lb_site.fit_transform([i.get('netloc') for i in df.url_features]))
            binary_frags = ('F', self.lb_frag.fit_transform([i.get('fragments') for i in df.url_features]))
            binary_queries = ('Q', self.mlb_query.fit_transform([i.get('queries') for i in df.url_features]))

            dataframes = []

            for i0 in (binary_sites, binary_frags, binary_queries):

                col_id = i0[0]
                binaries = i0[1]
                col_range = range(len(binaries[0]))
                column_names = ['{}{}'.format(col_id, bi) for bi in col_range]

                dataframes.append(pd.DataFrame(binaries, columns=column_names))

            self.df_url_features = pd.concat(dataframes, axis=1, sort=False)


        def encode_descriptors(self):

            binaries = self.mlb_descriptions.fit_transform(self.df_descriptors['descriptions'])
            column_names = ['{}{}'.format('D', i) for i in range(len(binaries[0]))]

            self.df_descriptor_features = pd.DataFrame(binaries, columns=column_names)


        def consolidate_feature_dataframes(self):

            df = pd.DataFrame({
                'site': self.classification_data['site'],
                'url': self.classification_data['url'],
                'height': self.classification_data['height'],
                'width': self.classification_data['width'],
                'links': self.classification_data['links'],
                'texts': self.classification_data['texts'],
                'images': self.classification_data['images'],
            })

            dataframes = [df, self.df_url_features, self.df_descriptor_features]

            self.primary_df = pd.concat(dataframes, axis=1, sort=False)


        def export_primary_dataframe(self):

            self.primary_df.to_csv('processed_urls.csv', index=None, header=True)


        def print_descriptors(self):
            """ print out most common occuring words in the dataset """

            word_dict = {}

            for lst in self.df_descriptors['descriptions']:
                for i in lst:
                    if not word_dict.get(i):
                        word_dict[i] = 1
                    else:
                        word_dict[i] += 1

            o = [(k.encode('utf-8'), v) for k,v in word_dict.items()]
            o.sort(key=lambda x: x[1], reverse=True)


    class learner():

        def __init__(self, selection=None):

            self.select_data(selection)


        def select_data(self, selection):

            input_data = pd.read_csv('processed_urls.csv')

            if selection:
                self.input_data = input_data[input_data['site'].isin(selection)]

            else:
                self.input_data = input_data


        def standardise_columns(self):
            print(self.input_data)


        def PCA(self):
            p = PCA(n_components=2)



    # convert_extract_to_csv(filename='page_classification_data.csv')

    # fobj = feature_extractor()
    # fobj.encode_url_features()
    # fobj.encode_descriptors()
    # fobj.consolidate_feature_dataframes()
    # fobj.export_primary_dataframe()
    # fobj.print_descriptors()

    lobj = learner(selection=['amazon'])
    lobj.standardise_columns()