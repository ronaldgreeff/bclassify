#*-*encoding: utf-8*-*
from peewee import *
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer, OneHotEncoder
import pandas as pd
import numpy as np
import json
import re
from urllib.parse import urlparse, parse_qs

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

    selection = Extract.select().where(Extract.extract.is_null(False))
    e = []
    for select in selection:

        site = select.site.lower()
        extract = json.loads(select.extract)

        d = {
            'site': extract['site'],
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

                elif k in ('og:type', ):
                    type_string = meta_tags[k]

                    for word in re.split(re_toke, type_string):
                        word = word.lower()
                        if word and (len(word)>1) and word.isalpha() and word != site:
                            types.add(word)


            if descriptions:
                d['descriptions'] = ','.join(descriptions)

            if titles:
                d['titles'] = ','.join(titles)

            if types:
                d['types'] = ','.join(types)

        else:

            d.update(
                {k:v for k, v in extract['meta_tags'].items()})

            d.update(
                {k:v for k, v in extract['body']['computed'].items()})

        e.append(d)

    df = pd.DataFrame(e)
    df.to_csv(filename, index=None, header=True)



if __name__ == '__main__':

    # convert_extract_to_csv(filename='page_classification_data.csv')
    # df = (pd.read_csv('page_classification_data.csv', engine='python'))
    # print(df.info())

    class feature_extractor():

        def __init__(self):
            self.classification_data = pd.read_csv("page_classification_data.csv")

            self.lb_site = LabelBinarizer()
            self.lb_frag = LabelBinarizer()
            self.mlb_query = MultiLabelBinarizer()
            # self.oe.path = OrdinalEncoder()

        def get_url_features(self, url):

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
            # url = 'https://www.asos.com/women/a-to-z-of-brands/adidas/cat/?cid=5906&refine=attribute_10992:61388&nlid=ww|shoes|shop+by+brand'
            # u = get_url_features(url)

        def encode_features(self):

            url_features = map(self.get_url_features, self.classification_data['url'])

            sites = []
            fragments = []
            queries = []

            for i in url_features:
                sites.append(i['netloc'])
                fragments.append(i['fragments'])
                queries.append(i['queries'])

            self.binary_sites = ('S', self.lb_site.fit_transform(sites))
            self.binary_frags = ('F', self.lb_frag.fit_transform(fragments))
            self.binary_queries = ('Q', self.mlb_query.fit_transform(queries))

        def create_dataframe(self):

            df = pd.DataFrame()
            df['url'] = self.classification_data['url']

            dataframes = [df,]

            for i0 in (self.binary_sites, self.binary_frags, self.binary_queries):

                col_id = i0[0]
                binaries = i0[1]
                col_range = range(len(binaries[0]))

                column_names = ['{}{}'.format(col_id, bi) for bi in col_range]
                column_values_set = []

                for i1 in range(len(binaries)):
                    row = binaries[i1]
                    column_values_set.append(row)

                    # column_values = []
                    # for i2 in range(len(row)):
                    #     column_values.append(row[i2])

                dataframes.append(pd.DataFrame(column_values_set, columns=column_names))

            self.primary_df = pd.concat(dataframes, axis=1, sort=False)

            # result = pd.concat([df2, df3], axis=1, sort=False)

            self.primary_df.to_csv('processed_urls.csv', index=None, header=True)


    fobj = feature_extractor()
    fobj.encode_features()
    fobj.create_dataframe()