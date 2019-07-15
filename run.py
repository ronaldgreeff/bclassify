#*-*encoding: utf-8*-*
from peewee import *
from sklearn.preprocessing import LabelEncoder
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


if __name__ == '__main__':

    def create_csv(filename, limited=True):

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


    # create_csv(filename='page_classification_data.csv')

    # df = (pd.read_csv('page_classification_data.csv', engine='python'))
    # print(df.info())

    def get_url_info(url):
        """ scheme://netloc/path;parameters?query#fragment

        https://www.amazon.co.uk/ap/signin?openid.return_to=https%3A%2F%2Fwww.amazon.co.uk%2Fref%3Dgw_sgn_ib%2F259-3956818-6697345&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=gbflex&openid.mode=checkid_setup&marketPlaceId=A1F83G8C2ARO7P&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&#collection
        ParseResult(
            scheme='https',
            netloc='www.amazon.co.uk',
            path='/ap/signin',
            params='',
            query='openid.return_to=https%3A%2F%2Fwww.amazon.co.uk%2Fref%3Dgw_sgn_ib%2F259-3956818-6697345&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=gbflex&openid.mode=checkid_setup&marketPlaceId=A1F83G8C2ARO7P&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&',
            fragment='collection')

        qs = {
            'openid.return_to': ['https://www.amazon.co.uk/ref=gw_sgn_ib/259-3956818-6697345'],
            'openid.identity': ['http://specs.openid.net/auth/2.0/identifier_select'],
            'openid.assoc_handle': ['gbflex'],
            'openid.mode': ['checkid_setup'],
            'marketPlaceId': ['A1F83G8C2ARO7P'],
            'openid.claimed_id': ['http://specs.openid.net/auth/2.0/identifier_select'],
            'openid.ns': ['http://specs.openid.net/auth/2.0']}
        """

        u = urlparse(url)

        string = ''.join(u[2:]) # excludes scheme & netloc

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

        d = {
            'length': len(string),
            'counts': {
                'lower': len(cu),
                'upper': len(cl),
                'numeric': len(n),
                'symbols' len(s)
            },
            'segments': len(u.path.split('/')),
            'queries': parse_qs(u.query).keys(),
            'fragments': u[5],
        }

        # queries = len(o.)

    url = 'https://www.amazon.co.uk/ap/signin?openid.return_to=https%3A%2F%2Fwww.amazon.co.uk%2Fref%3Dgw_sgn_ib%2F259-3956818-6697345&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=gbflex&openid.mode=checkid_setup&marketPlaceId=A1F83G8C2ARO7P&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&#t=40,80&xywh=160,120,320,240'
    get_url_info(url)

    # data['og:url'] = split_url(list(data['og:url']))

    # # description_encoder = LabelEncoder()
    # # ogtype_encoder = LabelEncoder()
    # # url_encoder = LabelEncoder()
