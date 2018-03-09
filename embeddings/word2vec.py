import random
from collections import namedtuple
from os import path

from gensim.models import KeyedVectors
from tqdm import tqdm
from embeddings.embedding import Embedding


class Word2VecEmbedding(Embedding):
    """
    Reference: https://arxiv.org/abs/1310.4546
    """

    Word2VecSetting = namedtuple('Word2VecSetting', ['url', 'extension', 'd_emb', 'size', 'description'])
    settings = {
        'google_news': Word2VecSetting('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz',
                                        'bin.gz', 300, 3000000, '3 million words and phrases'),
    }

    def __init__(self, name='google_news', show_progress=True):
        """

        Args:
            name: name of the embedding to retrieve.
            show_progress: whether to print progress.
        """
        assert name in self.settings, '{} is not a valid corpus. Valid options: {}'.format(name, self.settings)
        self.setting = self.settings[name]

        super().__init__()

        self.extension = self.setting.extension
        self.d_emb = self.setting.d_emb
        self.name = name
        self.db = self.initialize_db(self.path(path.join('word2vec', '{}:{}.db'.format(name, self.d_emb))))

        if len(self) < self.setting.size:
            self.clear()
            self.load_word2emb(show_progress=show_progress)

    def emb(self, word, default=lambda: None):
        return self.lookup(word, default)

    def load_word2emb(self, show_progress=True, batch_size=1000):
        fin_name = self.ensure_file(path.join('word2vec', '{}.{}'.format(self.name, self.extension)), url=self.setting.url)

        model = KeyedVectors.load_word2vec_format(fin_name, binary=('bin' in self.extension))
        vocabs = model.vocab.keys()
        if show_progress:
            vocabs = tqdm(vocabs)
        batch = []
        for word in vocabs:
            vec = model[word]
            batch.append((word, vec))
            if len(batch) == batch_size:
                self.insert_batch(batch)
                batch.clear()
        if batch:
            self.insert_batch(batch)


if __name__ == '__main__':
    from time import time
    emb = Word2VecSetting('google_news', show_progress=True)
    for w in ['canada', 'vancouver', 'toronto']:
        start = time()
        print('embedding {}'.format(w))
        print(emb.emb(w))
        print('took {}s'.format(time() - start))
