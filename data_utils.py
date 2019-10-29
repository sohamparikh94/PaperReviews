import json

class DataUtils:

    def group_by_revision(self, data):

        data_by_revision = dict()
        for doc_id in data:
            if('revisions' in data[doc_id]):
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


    def get_combined_reviews(self, data):

        data_by_combined = dict()
        decision_dict = {'Accept': 0, 'Minor Revision': 1, 'Major Revision': 2, 'Reject': 3}
        for doc_id in data:
            if('revisions' in data[doc_id]):
                combined_review = ""
                if('0' in data[doc_id]['revisions']):
                    if('combined_decision' in data[doc_id]['revisions']['0']):
                        combined_decision = data[doc_id]['revisions']['0']['combined_decision']
                        if(combined_decision not in data_by_combined and combined_decision in decision_dict):
                            data_by_combined[combined_decision] = list()
                        if('reviews' in data[doc_id]['revisions']['0']):
                            for review in data[doc_id]['revisions']['0']['reviews']:
                                review_obj = data[doc_id]['revisions']['0']['reviews'][review]
                                if('text' in review_obj):
                                    if(review_obj['text'].strip()):
                                        combined_review += " " + review_obj['text'].strip()
            if(combined_review.strip() and combined_decision in decision_dict):
                data_by_combined[combined_decision].append(combined_review)

        return data_by_combined



    def group_by_disagreement_multinomial(self, data):

        data_by_disagreement = dict()
        decision_dict = {'Accept': 0, 'Minor Revision': 1, 'Major Revision': 2, 'Reject': 3}
        for doc_id in data:
            if('revisions' in data[doc_id]):
                if('0' in data[doc_id]['revisions']):
                    if('combined_decision' in data[doc_id]['revisions']['0']):
                        combined_decision = data[doc_id]['revisions']['0']['combined_decision']
                        if(combined_decision not in data_by_disagreement):
                            data_by_disagreement[combined_decision] = list()
                        if('reviews' in data[doc_id]['revisions']['0']):
                            for review in data[doc_id]['revisions']['0']['reviews']:
                                review_obj = data[doc_id]['revisions']['0']['reviews'][review]
                                if('decision' in review_obj and 'text' in review_obj):
                                    if(review_obj['decision'] and review_obj['text'].strip() and review_obj['decision'].strip() in decision_dict):
                                        data_by_disagreement[combined_decision].append({'text': review_obj['text'], 'decision': decision_dict[review_obj['decision']]})

        return data_by_disagreement

    def group_by_disagreement_binary(self, data):

        data_by_disagreement = dict()
        data_by_disagreement['good'] = list()
        data_by_disagreement['bad'] = list()
        decision_dict = {'Accept': 0, 'Minor Revision': 1, 'Major Revision': 2, 'Reject': 3}
        for doc_id in data:
            if('revisions' in data[doc_id]):
                if('0' in data[doc_id]['revisions']):
                    if('combined_decision' in data[doc_id]['revisions']['0']):
                        combined_decision = data[doc_id]['revisions']['0']['combined_decision']
                        if('reviews' in data[doc_id]['revisions']['0']):
                            for review in data[doc_id]['revisions']['0']['reviews']:
                                if('decision' in data[doc_id]['revisions']['0']['reviews'][review]):
                                    decision = data[doc_id]['revisions']['0']['reviews'][review]['decision']
                                    if(decision):
                                        if('text' in data[doc_id]['revisions']['0']['reviews'][review]):
                                            if(data[doc_id]['revisions']['0']['reviews'][review]['text'].strip()):
                                                if(decision in decision_dict):
                                                    if(decision == combined_decision):
                                                        data_by_disagreement['good'].append({'text': data[doc_id]['revisions']['0']['reviews'][review]['text'], 'decision': decision_dict[decision]})
                                                    else:
                                                        data_by_disagreement['bad'].append({'text': data[doc_id]['revisions']['0']['reviews'][review]['text'], 'decision': decision_dict[decision]})

        return data_by_disagreement


    def load_decision_data(self):

        with open('../data/review_decisions.json') as f:
            data = json.load(f)
        data_by_revision = self.group_by_revision(data)
        documents = data_by_revision['0']['Accept'] + data_by_revision['0']['Minor Revision'] + data_by_revision['0']['Major Revision'] + data_by_revision['0']['Reject']
        labels = [0]*len(data_by_revision['0']['Accept']) + [1]*len(data_by_revision['0']['Minor Revision']) + [2]*len(data_by_revision['0']['Major Revision']) + [3]*len(data_by_revision['0']['Reject'])

        return documents, labels

    def load_disagreement_data(self, binary=True):

        with open('../data/review_decisions.json') as f:
            data = json.load(f)
        if(binary):
            data_by_disagreement = self.group_by_disagreement_binary(data)
            documents = data_by_disagreement['good'] + data_by_disagreement['bad']
            labels = [0]*len(data_by_disagreement['good']) + [1]*len(data_by_disagreement['bad'])
        else:
            data_by_disagreement = self.group_by_disagreement_multinomial(data)
            documents = data_by_disagreement['Accept'] + data_by_disagreement['Minor Revision'] + data_by_disagreement['Major Revision'] + data_by_disagreement['Reject']
            labels = [0]*len(data_by_disagreement['Accept'])+ [1]*len(data_by_disagreement['Minor Revision']) + [2]*len(data_by_disagreement['Major Revision']) + [3]*len(data_by_disagreement['Reject'])

        return documents, labels


    def load_combined_review_data(self):

        with open('../data/review_decisions.json') as f:
            data = json.load(f)
        data_by_combined = self.get_combined_reviews(data)
        documents = data_by_disagreement['Accept'] + data_by_disagreement['Minor Revision'] + data_by_disagreement['Major Revision'] + data_by_disagreement['Reject']
        labels = [0]*len(data_by_disagreement['Accept']) + [1]*len(data_by_disagreement['Minor Revision']) + [2]*len(data_by_disagreement['Major Revision']) + [3]*len(data_by_disagreement['Reject'])

        return documents, labels



    