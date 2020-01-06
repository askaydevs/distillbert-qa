from __future__ import absolute_import, division, print_function

import collections
import logging
import math

import numpy as np
import torch

from transformers import (WEIGHTS_NAME, DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)

from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from prediction_utils import (read_squad_examples, convert_examples_to_features, to_list, write_predictions)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


class Model:

    def __init__(self, path:str):
        self.max_seq_length = 384
        self.doc_stride = 128
        self.max_query_length = 64
        self.do_lower_case = True
        self.n_best_size = 20
        self.max_answer_length = 30
        self.eval_batch_size = 1
        self.model, self.tokenizer = self.model_load(path)
        self.model.eval()

    def model_load(self, path:str):
        config = DistilBertConfig.from_pretrained(path + "/config.json")
        tokenizer = DistilBertTokenizer.from_pretrained(path, do_lower_case=self.do_lower_case)
        model = DistilBertForQuestionAnswering.from_pretrained(path, from_tf=False, config=config)
        return model, tokenizer

    def predict(self, context, question):

        examples = read_squad_examples(context, question)
        features = convert_examples_to_features(examples, self.tokenizer, self.max_seq_length, self.doc_stride, self.max_query_length)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index)


        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.eval_batch_size)

        all_results = []
        for batch in (eval_dataloader):
            batch = tuple(t for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1]
                          }
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
                all_results.append(result)

        answer = write_predictions(examples, features, all_results, self.do_lower_case, self.n_best_size, self.max_answer_length)
        return answer
