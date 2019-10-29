import spacy
import string
import random
import numpy as np
from tqdm import tqdm
from scipy import sparse
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from IPython import embed
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class ClassifierUtils:

    def __init__(self, pretrained_dir):

        self.nlp_light = spacy.load('en', disable=['tagger', 'parser', 'ner'])
        self.nlp = spacy.load('en')
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.load_glove(pretrained_dir)
        self.alphabet = string.ascii_lowercase


    def load_glove(self, pretrained_dir):

        self.glove_embeddings = dict()
        with open(os.path.join(pretrained_dir, 'glove', 'glove.840B.300d.txt')) as f:
            print("Loading GloVe Embeddings")
            for line in tqdm(f):
                split_line = line.split()
                word = split_line[0]
                embedding = [float(x) for x in split_line[1:]]
                glove_embeddings[word] = embedding


    def forbidden_sw(self, token):

        if(token.lower() in self.stop_words):
            return True
        else:
            for character in token.lower():
                if(character in self.alphabet):
                    return False
            return True

        return False


    def forbidden(self, token):

        for character in token.lower():
            if(character in self.alphabet):
                return False

        return True


    def prepare_for_input(self, documents, label, vectorizer='count'):

        y = [label]*len(documents)
        if(vectorizer == 'count'):
            X = self.count_vectorizer(documents)
        elif(vectorizer == 'tfidf'):
            X = self.tfidf_vectorizer(documents)

        return X, y

    def form_single_split(self, docs, labels, ratio=0.8):

        indices = np.arange(len(docs))
        random.shuffle(indices)
        X_shuffled = [docs[i] for i in indices]
        y_shuffled = [labels[i] for i in indices]
        X_train = X_shuffled[:int(len(docs)*0.8)]
        y_train = y_shuffled[:int(len(docs)*0.8)]
        X_test = X_shuffled[int(len(docs)*0.8):]
        y_test = y_shuffled[int(len(docs)*0.8):]

        return X_train, X_test, y_train, y_test

    def form_splits(self, docs, labels, num_splits=5):

        indices = np.arange(len(docs))
        random.shuffle(indices)
        X_shuffled = [docs[i] for i in indices]
        y_shuffled = [labels[i] for i in indices]
        splits = list()

        for idx in range(num_splits):
            current_split = dict()
            start_index = int(idx*(len(X_shuffled)/num_splits))
            end_index = int((idx+1)*(len(X_shuffled)/num_splits))
            if(idx == num_splits - 1):
                current_split['test_docs'] = X_shuffled[start_index:max(end_index, len(X_shuffled))]
                current_split['y_test'] = y_shuffled[start_index:max(end_index, len(y_shuffled))]
                current_split['train_docs'] = X_shuffled[:start_index]
                current_split['y_train'] = y_shuffled[:start_index]
            else:
                current_split['test_docs'] = X_shuffled[start_index:end_index]
                current_split['y_test'] = y_shuffled[start_index:end_index]
                current_split['train_docs'] = X_shuffled[:start_index] + X_shuffled[end_index:]
                current_split['y_train'] = y_shuffled[:start_index] + y_shuffled[end_index:]

            splits.append(current_split)

        return splits


    def cross_validate(self, documents, labels, clf_metadata, features_metadata, num_splits=5):

        splits = self.form_splits(documents, labels, num_splits=num_splits)
        all_metrics = list()
        for split in splits:
            metrics = self.evaluate(split['train_docs'], split['y_train'], split['test_docs'], split['y_test'], clf_metadata, features_metadata)
            all_metrics.append(metrics)

        return all_metrics

    def get_classifier(self, clf_metadata):

        if(clf_metadata['type'] == 'NB'):
            clf = MultinomialNB()
        elif(clf_metadata['type'] == 'LR'):
            if(clf_metadata['multi_class'] == 'multinomial'):
                clf = LogisticRegression(multi_class='multinomial', solver='saga', n_jobs = clf_metadata['n_jobs'])
            else:
                clf = LogisticRegression(n_jobs=clf_metadata['n_jobs'])
        elif(clf_metadata['type'] == 'RF'):
            clf = RandomForestClassifier(n_estimators=clf_metadata['n_estimators'], max_depth=clf_metadata['max_depth'], n_jobs=clf_metadata['n_jobs'])
        else:
            raise NotImplementedError("Classifier type %s is not supported" % clf_metadata['type'])

        return clf

    def prepare_text_features(self, features_metadata, train_docs, test_docs):

        if(features_metadata['use_sw']):
            tokenizer = self.tokenizer
        else:
            tokenizer = self.tokenizer_sw
        if(features_metadata['type'] == 'count'):
            vectorizer = CountVectorizer(tokenizer=tokenizer, binary=features_metadata['binary'])
            vectorizer.fit(train_docs)
            X_train = vectorizer.transform(train_docs)
            X_test = vectorizer.transform(test_docs)
            if(features_metadata['use_length'] and not features_metadata['normalize']):
                X_train = sparse.csr_matrix(np.concatenate((X_train.toarray(), np.sum(X_train, axis=1)), axis=1))
                X_test = sparse.csr_matrix(np.concatenate((X_test.toarray(), np.sum(X_test, axis=1)), axis=1))
            elif(features_metadata['use_length'] and features_metadata['normalize'] and not features_metadata['binary']):
                lengths_train = np.sum(X_train, axis=1)
                lengths_test = np.sum(X_test, axis=1)
                X_train = X_train/lengths_train
                X_test = X_test/lengths_test
                X_train = sparse.csr_matrix(np.concatenate((X_train.toarray(), lengths_train), axis=1))
                X_test = sparse.csr_matrix(np.concatenate((X_test.toarray(), lengths_test), axis=1))
            elif(not features_metadata['binary'] and features_metadata['normalize'] and not features_metadata['append_binary']):
                X_train = X_train/np.sum(X_train, axis=1)
                X_test = X_test/np.sum(X_test, axis=1)
            elif(not features_metadata['binary'] and features_metadata['append_binary']):
                if(features_metadata['normalize']):
                    X_train = X_train/np.sum(X_train, axis=1)
                    X_test = X_test/np.sum(X_test,axis=1)
                else:
                    X_train = X_train.toarray()
                    X_test = X_test.toarray()
                train_bool = X_train > 0
                test_bool = X_test > 0
                train_bool = train_bool.astype(np.int64)
                test_bool = test_bool.astype(np.int64)
                X_train = sparse.csr_matrix(np.concatenate((X_train, train_bool), axis=1))
                X_test = sparse.csr_matrix(np.concatenate((X_test, test_bool), axis=1))
        elif(features_metadata['type'] == 'tfidf'):
            vectorizer = TfidfVectorizer(tokenizer=tokenizer)
            X_train = vectorizer.transform(train_docs)
            X_test = vectorizer.transform(test_docs)
        elif(features_metadata['type'] == 'glove_bow'):


        return X_train, X_test

    def one_hot(self, lst, num_classes = None):

        if(num_classes is None):
            num_classes = len(Counter(lst))
        encoding = np.zeros((len(lst), num_classes))
        encoding[np.arange(len(lst)), lst] = 1

        return encoding

    def encode_decision(self, txt_train, txt_test, train_docs, test_docs, decision_encoding):

        if(decision_encoding == 1):
            train_decision_features = self.one_hot([doc['decision'] for doc in train_docs])
            test_decision_features = self.one_hot([doc['decision'] for doc in test_docs])
            train_dense = txt_train.todense()
            test_dense = txt_test.todense()
            X_train = np.concatenate((train_dense, train_decision_features), axis=1)
            X_test = np.concatenate((test_dense, test_decision_features), axis=1)
            X_train = sparse.csr_matrix(X_train)
            X_test = sparse.csr_matrix(X_test)

        return X_train, X_test


    def prepare_features(self, features_metadata, train_docs, test_docs):

        if(type(train_docs[0]) == dict):
            tr_docs = [doc['text'] for doc in train_docs]
            tst_docs = [doc['text'] for doc in test_docs]
            txt_train, txt_test = self.prepare_text_features(features_metadata, tr_docs, tst_docs)
            if(features_metadata['decision_encoding']):
                X_train, X_test = self.encode_decision(txt_train, txt_test, train_docs, test_docs, features_metadata['decision_encoding'])
            else:
                X_train = txt_train
                X_test = txt_test
        else:
            X_train, X_test = self.prepare_text_features(features_metadata, train_docs, test_docs)

        return X_train, X_test

    # def get_confusion_matrix(self, train_docs, y_train, test_docs, y_test, clf_metadata, features_metadata):

        

    def evaluate(self, train_docs, y_train, test_docs, y_test, clf_metadata, features_metadata):

        clf = self.get_classifier(clf_metadata)
        X_train, X_test = self.prepare_features(features_metadata, train_docs, test_docs)
        if(features_metadata['sampling'] == 'over'):
            ros = RandomOverSampler(random_state=0)
            X_train, y_train = ros.fit_resample(X_train, y_train)
        elif(features_metadata['sampling'] == 'under'):
            rus = RandomUnderSampler(random_state=0)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        metrics = classification_report(y_test, y_predicted, output_dict=True)
        metrics['accuracy'] = accuracy_score(y_test, y_predicted)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_predicted)
        
        return metrics


    def get_nb_features(self, train_docs, labels):

        clf = MultinomialNB()
        count_vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        count_vectorizer.fit(train_docs)
        X_train = count_vectorizer.transform(train_docs)
        clf.fit(X_train, labels)

        features = count_vectorizer.get_feature_names()
        ordered_features = [[features[idx] for idx in array[::-1]] for array in np.argsort(clf.coef_, axis=1)]

        return ordered_features


    def get_token_counts(self, documents, labels):
        token_counts = dict()
        doc_counts = dict()
        all_token_counts = Counter()
        all_doc_counts = Counter()


        token_counts[0] = Counter()
        token_counts[1] = Counter()
        token_counts[2] = Counter()
        token_counts[3] = Counter()
        doc_counts[0] = Counter()
        doc_counts[1] = Counter()
        doc_counts[2] = Counter()
        doc_counts[3] = Counter()


        for idx, document in enumerate(tqdm(documents)):
            tokenized_doc = [token.text for token in self.nlp_light(document.lower())]
            current_counter = Counter()
            for token in tokenized_doc:
                token_counts[labels[idx]][token] += 1
                all_token_counts[token] += 1
                if(not current_counter[token]):
                    doc_counts[labels[idx]][token] += 1
                    all_doc_counts[token] += 1
                current_counter[token] += 1

        return all_doc_counts, all_token_counts


    def get_rf_features(self, train_docs, labels, num_estimators, max_depth, use_stopwords=True):

        clf = RandomForestClassifier(n_estimators=num_estimators, max_depth=max_depth, n_jobs=8)
        if(use_stopwords):
            tokenizer = self.tokenizer
        else:
            tokenizer = self.tokenizer_sw
        vectorizer = CountVectorizer(tokenizer=tokenizer)
        vectorizer.fit(train_docs)
        X_train = vectorizer.transform(train_docs)
        clf.fit(X_train, labels)
        feature_names = vectorizer.get_feature_names()
        feature_importances = clf.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]
        important_features = list()
        for idx in sorted_indices:
            if(feature_importances[idx] > 0):
                important_features.append(feature_names[idx])

        return important_features



    def get_lr_features(self, train_docs, labels, multi_class='ovr', use_stopwords=True):

        all_doc_counts, all_token_counts = self.get_token_counts(train_docs, labels)
        if(multi_class == 'ovr'):
            clf = LogisticRegression(n_jobs=8)
        else:
            clf = LogisticRegression(multi_class='multinomial', solver='saga', n_jobs=8)
        if(use_stopwords):
            tokenizer = self.tokenizer
        else:
            tokenizer = self.tokenizer_sw
        count_vectorizer = CountVectorizer(tokenizer=tokenizer)
        count_vectorizer.fit(train_docs)
        X_train = count_vectorizer.transform(train_docs)
        clf.fit(X_train, labels)

        features = count_vectorizer.get_feature_names()
        ordered_features1 = [[features[idx] for idx in array[::-1]] for array in np.argsort(clf.coef_, axis=1)]
        weighted_by_doc = [[array[idx]*all_doc_counts[features[idx]] for idx in range(len(array))] for array in clf.coef_]
        weighted_by_token = [[array[idx]*all_token_counts[features[idx]] for idx in range(len(array))] for array in clf.coef_]
        ordered_features2 = [[features[idx] for idx in array[::-1]] for array in np.argsort(weighted_by_doc, axis=1)]
        ordered_features3 = [[features[idx] for idx in array[::-1]] for array in np.argsort(weighted_by_token, axis=1)]

        return ordered_features1, ordered_features2, ordered_features3


    def tokenizer(self, text):
        return [token.text for token in self.nlp_light(text) if (not self.forbidden(token.text))]

    def tokenizer_sw(self, text):
        return [token.text for token in self.nlp_light(text) if (not self.forbidden_sw(token.text))]