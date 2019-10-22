import json
import pickle as pkl
import spacy
from tqdm import tqdm
from IPython import embed
from analysis_utils import AnalyzerUtils
from classifier_utils import ClassifierUtils
import warnings
from data_utils import DataUtils


def main():
    data_utils = DataUtils()
    clf_utils = ClassifierUtils()
    decision_documents, decision_labels = data_utils.load_decision_data()
    disagreement_documents, disagreement_labels = data_utils.load_disagreement_data()
    clf_metadata = {
                'type': 'RF',
                'n_estimators': 500,
                'max_depth': 128,
                'n_jobs': 8
            }
    features_metadata = {
                    'type': 'count',
                    'use_sw': True,
                    'use_length': False,
                    'binary': False,
                    'normalize': False,
                    'append_binary': False,
                    'sampling': None
            }

    metrics = clf_utils.cross_validate(disagreement_documents, disagreement_labels, clf_metadata, features_metadata, num_splits=5)

    embed()
    # with open('../outputs/lr_features_nosw.pkl', 'wb') as f:
    #     pkl.dump(features, f)
    


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()