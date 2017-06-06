import os
import sys
import pickle

module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.data_utils import ensure_dir
from utils.embedding_wrappers.glove import GloveEmbeddingWrapper
from utils.embedding_wrappers.one_hot import OneHotEmbeddingWrapper
import model_config

if __name__ == '__main__':
    embedding_wrapper = None
    if model_config.embedding_type == 'glove':
        embedding_wrapper = GloveEmbeddingWrapper()
    elif model_config.embedding_type == 'one_hot':
        embedding_wrapper = OneHotEmbeddingWrapper()
    ensure_dir(model_config.processed_data_path)
    embedding_wrapper.build_vocab(model_config.vocab_path)
    with open(model_config.vocab_path, 'w') as f:
        pickle.dump(embedding_wrapper.vocab, f)
        f.close()
    embedding_wrapper.process_embeddings(model_config.embeddings_path)
