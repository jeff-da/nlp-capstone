from typing import Dict
import json
import logging
import nltk

from overrides import overrides
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
def read_from_file(file_path):
    array = []
    with open(file_path) as ins:
        for line in ins:
            array.append(line)
    return array

@DatasetReader.register("maxent_reader")
class SemanticScholarDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.feature_map = {}
        data = read_from_file("/home/jzda/code/nlvr/mem_features/features_fixed.txt")
        idx = 0
        for line in data:
            tokens = line.split()
            for token in tokens:
                if "#" in token:
                    self.feature_map[token] = idx
                    idx = idx + 1

        self.id_to_features = {}
        for line in data:
            if len(line.split()) > 0:
                tokens = line.split()
                # print(tokens)
                id = tokens[0]
                self.id_to_features[id] = [0] * idx
                for token in tokens:
                    if "#" in token:
                        if token in self.feature_map:
                            self.id_to_features[id][self.feature_map[token]] = self.id_to_features[id][self.feature_map[token]] + 1

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                tokens = paper_json['sentence']
                label = paper_json['label']
                if 'directory' in paper_json:
                    id = { 'identifier': paper_json['identifier'], 'directory': paper_json['directory'] }
                else:
                    id = { 'identifier': paper_json['identifier'] }
                yield self.text_to_instance(tokens, id, label)

    @overrides
    def text_to_instance(self, tokens: str, metadata: Dict[str, str], label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        # tokenized_tokens = self._tokenizer.tokenize(tokens)
        # tokens_field = TextField(tokenized_tokens, self._token_indexers)
        tokens_field = ArrayField(np.array(self.id_to_features[metadata['identifier']]))

        fields = {'tokens': tokens_field, 'metadata': MetadataField(metadata)}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
