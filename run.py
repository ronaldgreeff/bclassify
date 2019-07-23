#*-*encoding: utf-8*-*
from peewee import *
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, MultiLabelBinarizer, OneHotEncoder
import pandas as pd
import numpy as np
import json
import re
from urllib.parse import urlparse, parse_qs
import nltk
from nltk.corpus import stopwords
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
            self.mlb_descriptions = MultiLabelBinarizer()
            self.mlb_titles = MultiLabelBinarizer()
            # self.oe.path = OrdinalEncoder()

            # self.stop_words = set(stopwords.words('english'))
            self.stop_words = ['a', "a's", 'able', 'about', 'above', 'abroad', 'according', 'accordingly', 'across', 'actually', 'adj', 'after', 'afterwards', 'again', 'against', 'ago', 'ahead', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'also', 'although', 'always', 'am', 'amid', 'amidst', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', "aren't", 'around', 'as', 'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully', 'b', 'back', 'backward', 'backwards', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'c', "c'mon", "c's", 'came', 'can', "can't", 'cannot', 'cant', 'caption', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'co', 'co.', 'com', 'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'course', 'currently', 'd', 'dare', "daren't", 'definitely', 'described', 'despite', 'did', "didn't", 'different', 'directly', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downwards', 'during', 'e', 'each', 'edu', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 'entirely', 'especially', 'et', 'etc', 'even', 'ever', 'evermore', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'fairly', 'far', 'farther', 'few', 'fewer', 'fifth', 'first', 'five', 'followed', 'following', 'follows', 'for', 'forever', 'former', 'formerly', 'forth', 'forward', 'found', 'four', 'from', 'further', 'furthermore', 'g', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings', 'h', 'had', "hadn't", 'half', 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how', "how's", 'howbeit', 'however', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'inc.', 'indeed', 'indicate', 'indicated', 'indicates', 'inner', 'inside', 'insofar', 'instead', 'into', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'l', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', "let's", 'like', 'liked', 'likely', 'likewise', 'little', 'look', 'looking', 'looks', 'low', 'lower', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', "mayn't", 'me', 'mean', 'meantime', 'meanwhile', 'merely', 'might', "mightn't", 'mine', 'minus', 'miss', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'must', "mustn't", 'my', 'myself', 'n', 'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', "needn't", 'needs', 'neither', 'never', 'neverf', 'neverless', 'nevertheless', 'new', 'next', 'nine', 'ninety', 'no', 'no-one', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'not', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'o', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', "one's", 'ones', 'only', 'onto', 'opposite', 'or', 'other', 'others', 'otherwise', 'ought', "oughtn't", 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own', 'p', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably', 'probably', 'provided', 'provides', 'q', 'que', 'quite', 'qv', 'r', 'rather', 'rd', 're', 'really', 'reasonably', 'recent', 'recently', 'regarding', 'regardless', 'regards', 'relatively', 'respectively', 'right', 'round', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'since', 'six', 'so', 'some', 'somebody', 'someday', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 't', "t's", 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'therefore', 'therein', 'theres', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'thing', 'things', 'think', 'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'till', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'twice', 'two', 'u', 'un', 'under', 'underneath', 'undoing', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'upwards', 'us', 'use', 'used', 'useful', 'uses', 'using', 'usually', 'v', 'value', 'various', 'versus', 'very', 'via', 'viz', 'vs', 'w', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", "we're", "we've", 'welcome', 'well', 'went', 'were', "weren't", 'what', "what'll", "what's", "what've", 'whatever', 'when', "when's", 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'whichever', 'while', 'whilst', 'whither', 'who', "who'd", "who'll", "who's", 'whoever', 'whole', 'whom', 'whomever', 'whose', 'why', "why's", 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'would', "wouldn't", 'x', 'y', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves', 'z', 'zero']

            self.get_descriptors()

        def get_descriptors(self):
            df = pd.DataFrame({
                'descriptions': self.classification_data['descriptions'] + ',' + self.classification_data['titles']
            }).fillna('empty')

            df['descriptions'] = df['descriptions'].apply(lambda x: [item for item in x.split(',') if item not in self.stop_words])

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

            url_features = map(self.extract_url_features, self.classification_data['url'])

            sites = []
            fragments = []
            queries = []

            for i in url_features:
                sites.append(i['netloc'])
                fragments.append(i['fragments'])
                queries.append(i['queries'])

            binary_sites = ('S', self.lb_site.fit_transform(sites))
            binary_frags = ('F', self.lb_frag.fit_transform(fragments))
            binary_queries = ('Q', self.mlb_query.fit_transform(queries))

            dataframes = []

            for i0 in (binary_sites, binary_frags, binary_queries):

                col_id = i0[0]
                binaries = i0[1]
                col_range = range(len(binaries[0]))
                column_names = ['{}{}'.format(col_id, bi) for bi in col_range]

                dataframes.append(pd.DataFrame(binaries, columns=column_names))

            self.df_url_features = pd.concat(dataframes, axis=1, sort=False)

        def add_specific_stopwords(self):
            """ print out most common occuring words in the dataset and add to self.stop_words """

            # print(self.df_descriptors)

            word_dict = {}

            for lst in self.df_descriptors['descriptions']:
                for i in lst:
                    if not word_dict.get(i):
                        word_dict[i] = 1
                    else:
                        word_dict[i] += 1

            sorted

            for i in word_dict:
                print(i.encode('utf-8'), word_dict[i])



        def encode_page_features(self):

            print(self.df_descriptors)

            # df['descriptions'] = df['descriptions'].apply(lambda x: [item for item in x.split() if item not in self.stop_words])

            # # print(self.stop_words)

            # pd.options.display.max_colwidth=1000
            # print(df.iloc[0])


        def create_features_dataframe(self):

            df = pd.DataFrame()

            df['url'] = self.classification_data['url']
            df['height'] = self.classification_data['height']
            df['width'] = self.classification_data['width']
            df['images'] = self.classification_data['images']
            df['links'] = self.classification_data['links']
            df['texts'] = self.classification_data['texts']

        def export_processed_features(self):

            self.primary_df.to_csv('test2.csv', index=None, header=True)


    fobj = feature_extractor()
    # fobj.encode_url_features()
    fobj.add_specific_stopwords()
    # fobj.encode_page_features()
    # fobj.create_features_dataframe()
    # fobj.create_features_dataframe()