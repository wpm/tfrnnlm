import itertools
import os
import pickle

import numpy as np
from tfrnnlm.text import IndexedVocabulary, PennTreebankTokenization, WhitespaceWordTokenization


def index_text_files(args):
    tokenization = {"word": WhitespaceWordTokenization(),
                    "penntb": PennTreebankTokenization()}[args.tokenization]
    # Create and save the vocabulary.
    if isinstance(tokenization, PennTreebankTokenization):
        # Penn Treebank data already contains an <unk> out of vocabulary symbol.
        out_of_vocabulary = "<unk>"
    else:
        out_of_vocabulary = None
    vocabulary_factory = IndexedVocabulary.factory(min_frequency=args.min_frequency, max_vocabulary=args.max_vocabulary,
                                                   out_of_vocabulary=out_of_vocabulary)
    documents = (open(document_name).read() for document_name in args.documents)
    vocabulary = vocabulary_from_documents(documents, tokenization, vocabulary_factory)
    with open(os.path.join(args.indexed_data_directory, "vocabulary"), "wb") as vocabulary_file:
        pickle.dump(vocabulary, vocabulary_file)
    # Use the vocabulary to index the files.
    for document_name in args.documents:
        with open(document_name) as document:
            indexed_document = np.array(list(vocabulary.index_tokens(document.read())))
            indexed_document_name = os.path.join(args.indexed_data_directory, os.path.basename(document_name))
            np.save(indexed_document_name, indexed_document)
    # Write information about the tokenization.
    info = [str(tokenization)]
    if args.min_frequency is not None:
        info.append("Minimum frequency %d" % args.min_frequency)
    if args.max_vocabulary is not None:
        info.append("Maximum vocabulary %d" % args.max_vocabulary)
    s = "\n".join(info) + "\n" + "\n".join(args.documents) + "\n"
    with open(os.path.join(args.indexed_data_directory, "info"), "w") as f:
        f.write(s)


def vocabulary_from_documents(documents, tokenization, vocabulary_factory):
    """
    Create an indexed vocabulary from a set of documents.

    :param documents: sequence of documents
    :type documents: iterable of str
    :param tokenization: document tokenizer
    :type tokenization: function str -> iterable of str
    :param vocabulary_factory: function to create a vocabulary given tokens
    :type vocabulary_factory: function iterable of str -> IndexedVocabulary
    :return: indexed vocabulary
    :rtype: IndexedVocabulary
    """
    tokens = itertools.chain(*(tokenization(document) for document in documents))
    return vocabulary_factory(tokens)
