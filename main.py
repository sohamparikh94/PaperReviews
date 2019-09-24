import json
import spacy
from IPython import embed
from analysis_utils import AnalyzerUtils

def main():

    analyzer = AnalyzerUtils()
    data = analyzer.load_review_data('../data/review_decisions.json')
    review_rating_pairs = analyzer.get_review_rating_pairs(data)
    word_counts = analyzer.get_classwise_word_count(review_rating_pairs)
    lengths = analyzer.get_classwise_lengths(review_rating_pairs)

    embed()


if __name__ == "__main__":
    main()