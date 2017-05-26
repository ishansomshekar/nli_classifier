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
    ensure_dir(model_config.processed_data_path)

    for ew, v_p, e_p in zip(model_config.get_embedding_wrappers(), model_config.get_vocab_paths(), model_config.get_embedding_paths()):
        ew.build_vocab(v_p)
        with open(v_p, 'w') as f:
            pickle.dump(ew.vocab, f)
            f.close()
        ew.process_embeddings(e_p)
