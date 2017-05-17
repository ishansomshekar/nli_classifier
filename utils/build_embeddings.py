import os
import sys
import pickle

module_home = os.environ['NLI_PATH']
sys.path.insert(0, module_home)

from utils.embedding_wrapper import EmbeddingWrapper

if __name__ == '__main__':
    embedding_wrapper = EmbeddingWrapper()
    vocab_path = embedding_wrapper.build_vocab()
    with open(vocab_path, 'w') as f:
        pickle.dump(embedding_wrapper.vocab, f)
        f.close()

    dict_obj = pickle.load(open(vocab_path, 'r'))

    embedding_wrapper.process_glove()
