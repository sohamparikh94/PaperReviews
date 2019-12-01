import os
import spacy
import string
import random
import numpy as np
from tqdm import tqdm
from IPython import embed
from scipy import sparse
from collections import Counter

from nltk import ngrams
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class ClassifierUtils:

    def __init__(self, pretrained_dir=None, load_glove=False, load_use=False):

        self.nlp_light = spacy.load('en', disable=['tagger', 'parser', 'ner'])
        self.nlp = spacy.load('en')
        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        if(load_glove):
            self.load_glove(pretrained_dir)
        self.alphabet = string.ascii_lowercase
        if(load_use):
            import tensorflow as tf
            import tensorflow_hub as hub
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()

            self.load_use_graph()


    def load_use_graph(self):

        graph = tf.Graph()
        with graph.as_default():
            self.text_input = tf.placeholder(tf.string, shape=[None])
            embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
            self.embedded_text = embed(self.text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        graph.finalize()
        self.session = tf.Session(graph=graph)
        self.session.run(init_op)



    def load_glove(self, pretrained_dir):

        self.glove_embeddings = dict()
        with open(os.path.join(pretrained_dir, 'glove', 'glove.840B.300d.txt')) as f:
            print("Loading GloVe Embeddings")
            for line in tqdm(f):
                split_line = line.split()
                n = len(split_line)
                word = ' '.join(split_line[0:(n-300+1)]).strip()
                embedding = [float(x) for x in split_line[-300:]]
                self.glove_embeddings[word] = embedding


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


    def cross_validate(self, documents, labels, clf_metadata, features_metadata, task='classification', num_splits=5):

        splits = self.form_splits(documents, labels, num_splits=num_splits)
        all_metrics = list()
        for split in splits:
            metrics = self.evaluate(split['train_docs'], split['y_train'], split['test_docs'], split['y_test'], clf_metadata, features_metadata, task=task)
            all_metrics.append(metrics)

        summarized_metrics = self.summarize_metrics(all_metrics, task)

        return summarized_metrics


    def summarize_metrics(self, all_metrics, task):

        summarized_metrics = dict()
        summarized_metrics['test'] = dict()
        summarized_metrics['train'] = dict()
        if(task == 'classification'):
            for label in all_metrics[0]['test']:
                if(len(label) > 1):
                    continue
                summarized_metrics['test'][label] = dict()
                summarized_metrics['train'][label] = dict()
                summarized_metrics['test'][label]['precision'] = np.mean([x['test'][label]['precision'] for x in all_metrics])
                summarized_metrics['test'][label]['recall'] = np.mean([x['test'][label]['recall'] for x in all_metrics])
                summarized_metrics['test'][label]['f1-score'] = np.mean([x['test'][label]['f1-score'] for x in all_metrics])
                summarized_metrics['train'][label]['precision'] = np.mean([x['train'][label]['precision'] for x in all_metrics])
                summarized_metrics['train'][label]['recall'] = np.mean([x['train'][label]['recall'] for x in all_metrics])
                summarized_metrics['train'][label]['f1-score'] = np.mean([x['train'][label]['f1-score'] for x in all_metrics])
            summarized_metrics['test']['accuracy'] = np.mean([x['test']['accuracy'] for x in all_metrics])
            summarized_metrics['train']['accuracy'] = np.mean([x['train']['accuracy'] for x in all_metrics])
            summarized_metrics['test']['macro avg'] = np.mean([x['test']['macro avg']['f1-score'] for x in all_metrics])
            summarized_metrics['train']['macro avg'] = np.mean([x['train']['macro avg']['f1-score'] for x in all_metrics])
            summarized_metrics['test']['weighted avg'] = np.mean([x['test']['weighted avg']['f1-score'] for x in all_metrics])
            summarized_metrics['train']['weighted avg'] = np.mean([x['train']['weighted avg']['f1-score'] for x in all_metrics])
        else:
            summarized_metrics['train']['mse'] = np.mean([x['train']['mse'] for x in all_metrics])
            summarized_metrics['train']['mae'] = np.mean([x['train']['mae'] for x in all_metrics])
            summarized_metrics['test']['mse'] = np.mean([x['test']['mse'] for x in all_metrics])
            summarized_metrics['test']['mae'] = np.mean([x['test']['mae'] for x in all_metrics])

        return summarized_metrics

    def get_classifier(self, clf_metadata):

        if(clf_metadata['type'] == 'NB'):
            clf = MultinomialNB()
        elif(clf_metadata['type'] == 'LR'):
            if(clf_metadata['multi_class'] == 'multinomial'):
                clf = LogisticRegression(multi_class='multinomial', solver='saga', n_jobs = clf_metadata['n_jobs'], penalty=clf_metadata['penalty'], C=clf_metadata['C'])
            else:
                clf = LogisticRegression(n_jobs=clf_metadata['n_jobs'], penalty=clf_metadata['penalty'], C=clf_metadata['C'])
        elif(clf_metadata['type'] == 'RF'):
            clf = RandomForestClassifier(n_estimators=clf_metadata['n_estimators'], max_depth=clf_metadata['max_depth'], n_jobs=clf_metadata['n_jobs'])
        elif(clf_metadata['type'] == 'OLS'):
            clf = LinearRegression()
        elif(clf_metadata['type'] == 'Ridge'):
            clf = Ridge(alpha=clf_metadata['alpha'])
        elif(clf_metadata['type'] == 'Lasso'):
            clf = Lasso(alpha=clf_metadata['alpha'])
        else:
            raise NotImplementedError("Classifier type %s is not supported" % clf_metadata['type'])

        return clf

    def prepare_text_features(self, features_metadata, train_docs, test_docs):

        if(features_metadata['use_sw']):
            tokenizer = self.tokenizer
        else:
            tokenizer = self.tokenizer_sw
        if(features_metadata['type'] == 'count'):
            vocabulary = self.prepare_vocabulary(train_docs, features_metadata)
            vectorizer = CountVectorizer(tokenizer=tokenizer, binary=features_metadata['binary'], vocabulary=vocabulary)
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
            X_train = self.embed_documents(train_docs, 'glove', tokenizer)
            X_test = self.embed_documents(test_docs, 'glove', tokenizer)
        elif(features_metadata['type'] == 'USE'):
            X_train = self.embed_documents(train_docs, 'USE', tokenizer)
            X_test = self.embed_documents(test_docs, 'USE', tokenizer)

        return X_train, X_test


    # def prepare_vocabulary(self, documents, restriction='word_count', min_frequency=0, max_frequency=float('inf'), min_df=0.0, max_df=1.0, min_dc = 0, max_dc = float('inf'), max_vocab_size=float('inf'), allow_stopwords=True):
    def prepare_vocabulary(self, documents, features_metadata):

        df = Counter()
        word_counts = Counter()
        vocabulary = list()


        if('min_frequency' in features_metadata):
            min_frequency = features_metadata['min_frequency']
        else:
            min_frequency = 0
        if('max_frequency' in features_metadata):
            max_frequency = features_metadata['max_frequency']
        else:
            max_frequency = float('inf')
        if('min_df' in features_metadata):
            min_df = features_metadata['min_df']
        else:
            min_df = 0
        if('max_df' in features_metadata):
            max_df = features_metadata['max_df']
        else:
            max_df = 1.0
        if('min_dc' in features_metadata):
            min_dc = features_metadata['min_dc']
        else:
            min_dc = 0
        if('max_dc' in features_metadata):
            max_dc = features_metadata['max_dc']
        else:
            max_dc = float('inf')
        if('max_vocab_size' in features_metadata):
            max_vocab_size = features_metadata['max_vocab_size']
        else:
            max_vocab_size = float('inf')
        
        
        if('ngram_range' in features_metadata):
            ngram_range = features_metadata['ngram_range']
        else:
            ngram_range = 1

        if(features_metadata['use_sw']):
            forbidden_fn = self.forbidden
        else:
            forbidden_fn = self.forbidden_sw

        for document in documents:
            curr_df = dict()
            tokens = [token.text for token in self.nlp_light(document.lower())]
            for word in tokens:
                word_counts[word] += 1
                curr_df[word] = 1
            for n in range(2, ngram_range+1):
                ngrams_iter = ngrams(tokens, n)
                for ngram in ngrams_iter:
                    word = ' '.join(ngram)
                    word_counts[word] += 1
                    curr_df[word] = 1
            for word in curr_df:
                df[word] += 1

        if(features_metadata['restriction'] == 'word_count'):
            for word in word_counts:
                if(not forbidden_fn(word)):
                    if(word_counts[word] > min_frequency and word_counts[word] < max_frequency):
                        vocabulary.append(word)
                        if(len(vocabulary) == max_vocab_size):
                            break
        elif(features_metadata['restriction'] == 'doc_frequency'):
            doc_count = len(documents)
            for word in df:
                df[word] = df[word]/doc_count
            for pair in df.most_common():
                if(not forbidden_fn(pair[0])):
                    if(pair[1] > min_df and pair[1] < max_df):
                        vocabulary.append(pair[0])
                        if(len(vocabulary) == max_vocab_size):
                            break
        elif(features_metadata['restriction'] == 'doc_count'):
            for pair in df.most_common():
                if(not forbidden_fn(pair[0])):
                    if(pair[1] > min_dc and pair[1] < max_dc):
                        vocabulary.append(pair[0])
                        if(len(vocabulary) == max_vocab_size):
                            break


        return vocabulary
                




    def embed_documents(self, docs, embedding_type, tokenizer):

        if(embedding_type=='glove'):
            doc_features = list()
            for doc in tqdm(docs):
                tokens = tokenizer(doc)
                for token in tokens:
                    doc_feature = np.zeros((300,))
                    token_count = 0
                    if(token in self.glove_embeddings):
                        doc_feature += self.glove_embeddings[token]
                        token_count += 1
                doc_features.append(doc_feature)
            doc_features = np.array(doc_features)
        elif(embedding_type=='USE'):
            doc_features = list()
            for doc in docs:
                embedding = self.session.run(self.embedded_text, feed_dict={self.text_input: [doc]})
                doc_features.append(embedding[0])
            doc_features = np.array(doc_features)

        return doc_features


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

    def undersample(self, X_train, y_train):

        num_labels = Counter(y_train)
        examples = dict()
        undersampled_examples = dict()
        undersampled_X_train = list()
        undersampled_y_train = list()
        min_num = min([num_labels[x] for x in num_labels])
        for label in num_labels:
            examples[label] = list()
        for idx, ex in enumerate(X_train):
            examples[y_train[idx]].append(ex)
        for label in examples:
            undersampled_examples[label] = random.sample(examples[label], min_num)
        for label in examples:
            undersampled_X_train += undersampled_examples[label]
            undersampled_y_train += [label]*min_num

        return undersampled_X_train, undersampled_y_train


    def oversample(self, X_train, y_train):

        num_labels = Counter(y_train)
        examples = dict()
        oversampled_examples = dict()
        oversampled_X_train = list()
        oversampled_y_train = list()
        max_num = max([num_labels[x] for x in num_labels])
        for label in num_labels:
            examples[label] = list()
        for idx, ex in enumerate(X_train):
            examples[y_train[idx]].append(ex)
        for label in examples:
            oversampled_examples[label] = list()
            for _ in range(max_num):
                oversampled_examples[label].append(random.choice(examples[label]))
            # oversampled_examples[y_train[idx]] = random.sample(examples[y_train[idx]], max_num)
        for label in examples:
            oversampled_X_train += oversampled_examples[label]
            oversampled_y_train += [label]*max_num

        return oversampled_X_train, oversampled_y_train


    def get_metrics(self, clf, y_train, y_test, train_predicted, test_predicted, task):

        if(task == 'classification'):
            metrics = dict()
            metrics['test'] = classification_report(y_test, test_predicted, output_dict=True)
            metrics['test']['accuracy'] = accuracy_score(y_test, test_predicted)
            metrics['test']['confusion_matrix'] = confusion_matrix(y_test, test_predicted)
            metrics['train'] = classification_report(y_train, train_predicted, output_dict=True)
            metrics['train']['accuracy'] = accuracy_score(y_train, train_predicted)
            metrics['train']['confusion_matrix'] = confusion_matrix(y_train, train_predicted)
        else:
            metrics = dict()
            metrics['test'] = dict()
            metrics['test']['mse'] = mean_squared_error(y_test, test_predicted)
            metrics['test']['mae'] = mean_absolute_error(y_test, test_predicted)
            metrics['train']['mse'] = mean_squared_error(y_train, train_predicted)
            metrics['train']['mae'] = mean_absolute_error(y_train, train_predicted)

        return metrics


    def evaluate(self, train_docs, y_train, test_docs, y_test, clf_metadata, features_metadata, task='classification', return_predictions = False):

        clf = self.get_classifier(clf_metadata)
        X_train, X_test = self.prepare_features(features_metadata, train_docs, test_docs)
        if(features_metadata['sampling'] == 'over'):
            ros = RandomOverSampler(random_state=0)
            X_train, y_train = ros.fit_resample(X_train, y_train)
            # X_train, y_train = self.oversample(X_train, y_train)
            embed()
        elif(features_metadata['sampling'] == 'under'):
            rus = RandomUnderSampler(random_state=0)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            # X_train, y_train = self.undersample(X_train, y_train)
        clf.fit(X_train, y_train)
        test_predicted = clf.predict(X_test)
        train_predicted = clf.predict(X_train)
        
        metrics = self.get_metrics(clf, y_train, y_test, train_predicted, test_predicted, task)

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