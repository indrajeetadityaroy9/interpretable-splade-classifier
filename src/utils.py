import nltk
from typing import Set, List, Optional

def load_stopwords() -> Set[str]:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

def simple_tokenizer(text: str, stopwords_set: Optional[Set[str]] = None) -> List[str]:
    """
    Simple whitespace tokenizer + lowercase + optional stopword removal.
    """
    tokens = text.lower().split()
    if stopwords_set:
        return [t for t in tokens if t not in stopwords_set]
    return tokens
