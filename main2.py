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

def main():

    clf_utils = ClassifierUtils()
    with open('../data/review_decisions.json') as f:
        data = json.load(f)
    data_by_revision = group_by_revision(data)
    documents = data_by_revision['0']['Reject'] + data_by_revision['0']['Accept'] + data_by_revision['0']['Major Revision'] + data_by_revision['0']['Minor Revision']
    labels = [0]*len(data_by_revision['0']['Reject']) + [1]*len(data_by_revision['0']['Accept']) + [2]*len(data_by_revision['0']['Major Revision']) + [3]*len(data_by_revision['0']['Minor Revision'])
    

    selected_metrics = dict()
    try:
        metrics1 = clf_utils.cross_validate(documents, labels, num_splits=5, clf_type='LR', multi_class='multinomial', normalize=False, binary=False, n_estimators=50, max_depth=64, use_length=False, append_binary=False)
    except:
        embed()
    try:
        metrics2 = clf_utils.cross_validate(documents, labels, num_splits=5, clf_type='LR', multi_class='multinomial', normalize=False, binary=False, n_estimators=50, max_depth=64, use_length=True, append_binary=False)
    except:
        embed()
    try:
        metrics3 = clf_utils.cross_validate(documents, labels, num_splits=5, clf_type='LR', multi_class='multinomial', normalize=False, binary=True, n_estimators=50, max_depth=64, use_length=False, append_binary=False)
    except:
        embed()
    try:
        metrics4 = clf_utils.cross_validate(documents, labels, num_splits=5, clf_type='LR', multi_class='multinomial', normalize=True, binary=False, n_estimators=50, max_depth=64, use_length=False, append_binary=False)
    except:
        embed()
    metrics = dict()
    metrics['1'] = metrics1
    metrics['2'] = metrics2
    metrics['3'] = metrics3
    metrics['4'] = metrics4
    with open('../outputs/lr_multi_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    # metrics5 = clf_utils.cross_validate(documents, labels, num_splits=5, clf_type='LR', normalize=False, binary=False, n_estimators=50, max_depth=64, use_length=False, append_binary=True)
    # metrics6 = clf_utils.cross_validate(documents, labels, num_splits=5, clf_type='LR', normalize=True, binary=False, n_estimators=50, max_depth=64, use_length=False, append_binary=True)

    embed()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()