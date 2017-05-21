import os
import sys
import pickle

module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.data_utils import ensure_dir

from utils.embedding_wrappers.glove import GloveEmbeddingWrapper
from utils.embedding_wrappers.one_hot import OneHotEmbeddingWrapper
from utils.embedding_wrappers.pos import PosEmbeddingWrapper

import model_config

if __name__ == '__main__':
    pos_embedding_wrapper = PosEmbeddingWrapper()
    word_embedding_wrapper = model_config.get_embedding_wrapper()
    ensure_dir(model_config.processed_data_path)

    pos_embedding_wrapper.build_vocab(model_config.pos_path)
    with open(model_config.pos_path, 'w') as f:
        pickle.dump(pos_embedding_wrapper.vocab, f)
        f.close()
    pos_embedding_wrapper.process_embeddings(model_config.pos_embeddings_path)

    word_embedding_wrapper.build_vocab(model_config.vocab_path)
    with open(model_config.vocab_path, 'w') as f:
        pickle.dump(word_embedding_wrapper.vocab, f)
        f.close()
    word_embedding_wrapper.process_embeddings(model_config.embeddings_path)
