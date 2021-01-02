import pandas as pd
import gensim as gs
import multiprocessing
from gensim.models.doc2vec import Doc2Vec

documents = []

 for index,row in articles.iterrows():
documents.append(gs.models.doc2vec.TaggedDocument(words=row['text'].split(),tags=['SENT_%s' % row['id']]))

cores = multiprocessing.cpu_count()

#### Parameters:\n",
   # "1. **dm** (int {1,0}) – Defines the training algorithm. If dm=1, ‘distributed memory’ (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
   # "2. **dbow_words** (int {1,0}) – If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
   # "3. **size** (int) – Dimensionality of the feature vectors.
    #"4. **window** (int) – The maximum distance between the current and predicted word within a sentence.
    #"5. **min_count** (int) – Ignores all words with total frequency lower than this.\n",
   # "6. **workers** (int) – Use these many worker threads to train the model (=faster training with multicore machines)."
### Some Notes about choosing the values of the Doc2Vec parameters:\n",
    #When you increase the value of the window, the duration of training is going to be longer (but usually with better results).\n",
    #Not just **window** can affect the duration of training. Also, the **size** can obviously have its effect on the duration of training.\n",
    #You have also not to forget the more you increase those values the more computational power you they are going to need."