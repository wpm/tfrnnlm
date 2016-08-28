import os

from tfrnnlm.text import PartitionedData, Vocabulary, WordTokenizer


def create_partitioned_data():
    return PartitionedData.from_text({
        "train": ["blue blue green", "red red red"],
        "test": ["green green red black"],
        "validate": ["red blue blue orange"]
    }, ["train"], Vocabulary.factory(WordTokenizer(True)))


def create_serialized_partitioned_data(directory):
    os.makedirs(directory)
    partitioned_data = create_partitioned_data()
    partitioned_data.serialize(directory)
    return partitioned_data
