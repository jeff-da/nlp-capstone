{

  "train_data_path": "/home/jzda/nlvr/nlvr2/data/train.json",
  "validation_data_path": "/home/jzda/nlvr/nlvr2/data/dev.json",

  "dataset_reader": {
    "type": "nlvr_reader",
    "token_indexers": {
       "tokens": {
         "type": "single_id",
         "lowercase_tokens": true
       }
    }
  },

  "model": {
    "type": "nlvr_classifier",
    "text_field_embedder": {
      "token_embedders": {
         "tokens": {
           "type": "embedding",
           "embedding_dim": 100,
           "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
           "trainable": true
         }
      }
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 2,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 200,
      "num_layers": 2,
      "hidden_dims": [100, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },

  "iterator": {
    "type": "basic",
    "batch_size": 1
  },

  "trainer": {
    "num_epochs": 5,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
