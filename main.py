import json
import spacy
from tqdm import tqdm
from IPython import embed
from analysis_utils import AnalyzerUtils
from classifier_utils import ClassifierUtils
import warnings

def group_by_revision(data):

    data_by_revision = dict()
    for doc_id in data:
        if('revisions') in data[doc_id]:
            for revision in data[doc_id]['revisions']:
                if(revision not in data_by_revision):
                    data_by_revision[revision] = dict()
                if('reviews' in data[doc_id]['revisions'][revision]):
                    for review in data[doc_id]['revisions'][revision]['reviews']:
                        review_obj = data[doc_id]['revisions'][revision]['reviews'][review]
                        if('text' in review_obj and 'decision' in review_obj):
                            if(len(review_obj['text'].strip()) > 0 and len(review_obj['decision']) > 0):
                                if(review_obj['decision'] not in data_by_revision[revision]):
                                    data_by_revision[revision][review_obj['decision']] = list()
                                data_by_revision[revision][review_obj['decision']].append(review_obj['text'])

    return data_by_revision

def load_data():
    with open('../data/review_decisions.json') as f:
        data = json.load(f)
    data_by_revision = group_by_revision(data)
    documents = data_by_revision['0']['Accept'] + data_by_revision['0']['Minor Revision'] + data_by_revision['0']['Major Revision'] + data_by_revision['0']['Reject']
    labels = [0]*len(data_by_revision['0']['Accept']) + [1]*len(data_by_revision['0']['Minor Revision']) + [2]*len(data_by_revision['0']['Major Revision']) + [3]*len(data_by_revision['0']['Reject'])

    return documents, labels

def main():

    clf_utils = ClassifierUtils()
    documents, labels = load_data()
    """
    clf_metadata = {'type': 'LR',
                    'multi_class': 'ovr'
    }
    """
    clf_metadata = {
        'type': 'RF',
        'n_estimators': 500,
        'max_depth': 128,
        'n_jobs': 1
    }

    features_metadata1 = {'type': 'count',
                        'use_sw': True,
                        'use_length': False,
                        'binary': False,
                        'normalize': False,
                        'append_binary': False,
                        'sampling': 'over'
    }
    features_metadata2 = {'type': 'count',
                        'use_sw': True,
                        'use_length': False,
                        'binary': False,
                        'normalize': False,
                        'append_binary': False,
                        'sampling': 'under'
    }
    # ordered_features1, ordered_features2, ordered_features3 = clf_utils.get_lr_features(documents, labels)
    metrics1 = clf_utils.cross_validate(documents, labels, clf_metadata, features_metadata1, num_splits=5)
    metrics2 = clf_utils.cross_validate(documents, labels, clf_metadata, features_metadata2, num_splits=5)

    
    embed()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()