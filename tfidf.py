import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

document1 = tb("شهرزاد وشهريار يحكي في قديم الزمان أنه عاش ملكا من ملوك ساسان وكان له ولدان ، الكبير يدعى شهريار والصغير يدعى شاه الزمان ، كان شهريار قد حكم ساسان بعد وفاة والده أما شاه الزمان فقد حكم سمرقند")

document2 = tb("كما أخبره بما حدث في قصره ، غضب شهريار ، وقطع رأس زوجته كما قطع رأس جميع الجواري والعبيد الموجودين بالقصر ، وأصبح الملك شهريار كل يوم يتزوج بنتًا بكرا من المدينة وفي اليوم التالي يقتلها ، وبقي على ذلك الحال ثلاثة سنوات ، فهربت الناس من المدينة ببناتها")
document3 = tb("مما منعه من إتمام خطته والسفر إلى فيينا، عاصمة الموسيقى في ذلك العصر.")

bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("Word: {}, TF-IDF: {}".format(word, round(score, 5)))
