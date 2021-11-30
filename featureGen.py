import numpy as np
import torch,os
import torch.nn as nn
#from utils import get_project_root
from model_for_feature import L1,L2,OrdinalRegression,BilinearContactMap
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

class Alphabet:
    def __init__(self, chars, encoding=None, mask=False, missing=255):
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + missing
        if encoding is None:
            self.encoding[self.chars] = np.arange(len(self.chars))
            self.size = len(self.chars)
        else:
            self.encoding[self.chars] = encoding
            self.size = encoding.max() + 1
        self.mask = mask
        if mask:
            self.size -= 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x):
        """ encode a byte string into alphabet indices """
        x = np.frombuffer(x, dtype=np.uint8)
        return self.encoding[x]

    def decode(self, x):
        """ decode index array, x, to byte string of this alphabet """
        string = self.chars[x]
        return string.tobytes()

    def unpack(self, h, k):
        """ unpack integer h into array of this alphabet with length k """
        n = self.size
        kmer = np.zeros(k, dtype=np.uint8)
        for i in reversed(range(k)):
            c = h % n
            kmer[i] = c
            h = h // n
        return kmer

    def get_kmer(self, h, k):
        """ retrieve byte string of length k decoded from integer h """
        kmer = self.unpack(h, k)
        return self.decode(kmer)


class Uniprot21(Alphabet):
    def __init__(self, mask=False):
        chars = alphabet = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11,4,20,20] # encode 'OUBZ' as synonyms
        super(Uniprot21, self).__init__(chars, encoding=encoding, mask=mask, missing=20)


def embed_sequence(model, x, pool='none', use_cuda=False,device = None):
    #device = torch.device('cuda:1')
    if len(x) == 0:
        n = model.embedding.proj.weight.size(1)
        z = np.zeros((1,n), dtype=np.float32)
        return z

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        x = x.to(device)

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = model.transform(x)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        #z = z.cpu().numpy()

    return z


class ProSEMT(nn.Module):
    def __init__(self, embedding, scop_predict, cmap_predict):
        super(ProSEMT, self).__init__()
        self.embedding = embedding
        self.scop_predict = scop_predict
        self.cmap_predict = cmap_predict

    @staticmethod
    def load_pretrained(path='prose_mt'):
        if path is None or path == 'prose_mt':
            root = get_project_root()
            path = os.path.join(root, 'Struct2Go/saved_models', 'prose_mt_3x1024.sav')

        from model_for_feature import SkipLSTM
        encoder = SkipLSTM(21, 21, 1024, 3)
        encoder.cloze = encoder.proj

        proj_in = encoder.proj.in_features
        proj = nn.Linear(proj_in, 100)
        encoder.proj = proj
        encoder.nout = 100

        scop_predict = OrdinalRegression(100, 5, compare=L1(), allow_insertions=False)
        cmap_predict = BilinearContactMap(proj_in)
        model = ProSEMT(encoder, scop_predict, cmap_predict)

        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def clip(self):
        self.scop_predict.clip()
        self.cmap_predict.clip()

    def forward(self, x):
        return self.embedding(x)

    def transform(self, x):
        return self.embedding.transform(x)

    def score(self, z_x, z_y):
        return self.scop_predict(z_x, z_y)

    def predict(self, z):
        return self.cmap_predict(z)
