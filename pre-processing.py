

# Tokenizing (Document to list of sentences. Sentence to list of words.)
def tokenize(txt):
    '''Tokenizes into sentences, then strips punctuation/abbr, converts to lowercase and tokenizes words'''
    return [word_tokenize(" ".join(re.findall(r'\w+', t,flags = re.UNICODE | re.LOCALE)).lower()) 
                for t in sent_tokenize(txt.replace("'", ""))]

#Removing stopwords. Takes list of words, outputs list of words.
def remove_stopwords(l_words, lang='english'):
    l_stopwords = stopwords.words(lang)
    content = [w for w in l_words if w.lower() not in l_stopwords]
    return content

#Stem all words with stemmer of type, return encoded as "encoding"
def stemming(words_l, type="PorterStemmer", lang="english", encoding="utf8"):
    supported_stemmers = ["PorterStemmer","SnowballStemmer","LancasterStemmer","WordNetLemmatizer"]
    if type is False or type not in supported_stemmers:
        return words_l
    else:
        l = []
        if type == "PorterStemmer":
            stemmer = PorterStemmer()
            for word in words_l:
                l.append(stemmer.stem(word).encode(encoding))
        if type == "SnowballStemmer":
            stemmer = SnowballStemmer(lang)
            for word in words_l:
                l.append(stemmer.stem(word).encode(encoding))
        if type == "LancasterStemmer":
            stemmer = LancasterStemmer()
            for word in words_l:
                l.append(stemmer.stem(word).encode(encoding))
        if type == "WordNetLemmatizer":
            wnl = WordNetLemmatizer()
            for word in words_l:
                l.append(wnl.lemmatize(word).encode(encoding))
        return l

def preprocess_pipeline(txt, lang="english", stemmer_type="WordNetLemmatizer"):
    l = []
    words = []
    
    sentences = tokenize(txt)
    for sentence in sentences:
        words = remove_stopwords(sentence, lang)
        words = stemming(words, stemmer_type)
        l.append(" ".join(words))
        return " ".join(l)
        
