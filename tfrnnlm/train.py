from tfrnnlm import logger
from tfrnnlm.text import whitespace_word_tokenization, IndexedVocabulary


def train_model(args):
    tokens = whitespace_word_tokenization(args.text.read())
    vocabulary = IndexedVocabulary(tokens)
    logger.info(vocabulary)
