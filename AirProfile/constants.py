from nltk.corpus import stopwords

LIWC_CATEGORY_KEYS = [
    'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they',
    'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future',
    'adverb', 'preps', 'conj', 'negate', 'quant', 'number', 'swear',
    'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo',
    'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep',
    'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear',
    'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ',
    'motion', 'space', 'time', 'work', 'achieve', 'leisure', 'home',
    'money', 'relig', 'death', 'assent', 'nonfl', 'filler'
]

LIWC_META_KEYS = ['WC', 'WPS', 'Sixltr', 'Dic', 'Numerals']

LIWC_PUNCT_KEYS = [
    'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash',
    'Quote', 'Apostro', 'Parenth', 'OtherP', 'AllPct'
]

LIWC_PUNCT = [('Period', '.'), ('Comma', ','), ('Colon', ':'),
              ('SemiC', ';'), ('QMark', '?'), ('Exclam', '!'),
              ('Dash', '-'), ('Quote', '"'), ('Apostro', "'"),
              ('Parenth', '()[]{}'), ('OtherP', '#$%&*+-/<=>@\\^_`|~')]

FEAT_WC_CATEGORIES = [
    'interestsTastes', 'lifeMottoValues', 'workEducation', 'relationships',
    'personality', 'originResidence', 'travel', 'hospitality'
]

FEAT_WC_LING = [
    'wc_log', 'readability', 'WPS', 'Apostro', 'Colon', 'Comma', 'Dash',
    'Exclam', 'Parenth', 'Period', 'QMark', 'Quote', 'SemiC', 'i', 'we',
    'you', 'shehe', 'they', 'ipron', 'article', 'auxverb', 'past',
    'present', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant',
    'number', 'swear', 'assent', 'nonfl', 'filler'
]

FEAT_WC_PSYCH = [
    'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo',
    'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep',
    'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear',
    'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ',
    'motion', 'space', 'time'
]

FEAT_WC_CONCERN = [
    'work', 'achieve', 'leisure', 'home', 'money', 'relig', 'death'
]

# Reverse LUT for efficiently filtering out non-alphanumeric characters.
NON_ALNUM_CHARS = ''.join(
    c for c in map(chr, xrange(256)) if not c.isalnum() and not c.isspace())

# TODO(kenlimmj): This should really be converted to a Trie for more
# efficient lookups.
STOPWORDS = set(
    stopwords.words('english') + [
        'all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway',
        'fifty', 'four', 'not', 'own', 'through', 'yourselves', 'go',
        'where', 'mill', 'only', 'find', 'before', 'one', 'whose',
        'system', 'how', 'somewhere', 'with', 'thick', 'show', 'had',
        'enough', 'should', 'to', 'must', 'whom', 'seeming', 'under',
        'ours', 'has', 'might', 'thereafter', 'latterly', 'do', 'them',
        'his', 'around', 'than', 'get', 'very', 'de', 'none', 'cannot',
        'every', 'whether', 'they', 'front', 'during', 'thus', 'now',
        'him', 'nor', 'name', 'several', 'hereafter', 'always', 'who',
        'cry', 'whither', 'this', 'someone', 'either', 'each', 'become',
        'thereupon', 'sometime', 'side', 'two', 'therein', 'twelve',
        'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up',
        'namely', 'towards', 'are', 'further', 'beyond', 'ourselves',
        'yet', 'out', 'even', 'will', 'what', 'still', 'for', 'bottom',
        'mine', 'since', 'please', 'forty', 'per', 'its', 'everything',
        'behind', 'un', 'above', 'between', 'it', 'neither', 'seemed',
        'ever', 'across', 'she', 'somehow', 'be', 'we', 'full', 'never',
        'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon',
        'nowhere', 'although', 'found', 'alone', 're', 'along', 'fifteen',
        'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many',
        'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount',
        'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already',
        'co', 'afterwards', 'formerly', 'within', 'seems', 'into',
        'others', 'while', 'whatever', 'except', 'down', 'hers',
        'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
        'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from',
        'her', 'few', 'together', 'top', 'there', 'due', 'been', 'next',
        'anyone', 'eleven', 'much', 'call', 'therefore', 'interest',
        'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 'empty',
        'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am',
        'becoming', 'hereby', 'amongst', 'else', 'part', 'everywhere',
        'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made',
        'twenty', 'these', 'bill', 'cant', 'us', 'until', 'besides',
        'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'your',
        'toward', 'my', 'something', 'and', 'whereafter', 'whenever',
        'give', 'almost', 'wherever', 'is', 'describe', 'beforehand',
        'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem',
        'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby',
        'thin', 'no', 'perhaps', 'latter', 'meanwhile', 'when', 'detail',
        'same', 'wherein', 'beside', 'also', 'that', 'other', 'take',
        'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may',
        'after', 'upon', 'most', 'hereupon', 'eight', 'but', 'serious',
        'nothing', 'such', 'why', 'a', 'off', 'whereby', 'third', 'i',
        'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours',
        'their', 'rather', 'without', 'so', 'five', 'the', 'first',
        'whereas', 'once'
    ])
