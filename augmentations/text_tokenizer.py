import nltk
import emoji
import re
import string
from tqdm.notebook import tqdm
nltk.download('stopwords')
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


class TextTokenizer():
    """
        Preprocess an input sentence, output new sentence

        *preprocess_steps: list
            None : Normal split sentence by space
            "base" : consists of to lower, remove punctuations, remove stopwords
            "stem" : Snollball Stem words in sentence
            "lemmatize" : Lemmatize words in sentence
            "remove_emojis" : Remove emojis from sentence
            "ngrams" : Add bigrams, trigrams to output tokens
            "replace_consecutive" : Trim off consecutive part in sentence
            

    """
    def __init__(self, preprocess_steps = None, max_length = None):

        self.max_length = max_length
        if preprocess_steps is not None:
            assert isinstance(preprocess_steps, list) , "preprocess_steps must be a list contains name of methods"
        self.preprocess_steps = preprocess_steps
        
        if preprocess_steps is not None:
            if "base" in preprocess_steps:
                self.punctuations = string.punctuation
                self.stopwords_list = stopwords.words("english")
            if "stem" in preprocess_steps:
                self.stemmer = SnowballStemmer('english')
            if "lemmatize" in preprocess_steps:
                self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, sentence):
        tokens = sentence.split()
        if self.preprocess_steps is not None:
            tokens = self.clean(tokens, self.preprocess_steps)
        if self.max_length is not None:
            tokens = tokens[:self.max_length]

        return tokens

    def remove_stopwords(self, tokens):
        if tokens in self.stopwords_list:
            return ''
        else:
            return tokens
    
    def add_n_grams(self, tokens):
        l = []
        if "2grams" in self.preprocess_steps or "ngrams" in self.preprocess_steps:
            bigrams = ngrams(tokens, 2)
            for i in bigrams:
                l.append(" ".join(i))
        if "3grams" in self.preprocess_steps or "ngrams" in self.preprocess_steps:
            trigrams = ngrams(tokens, 3)
            for j in trigrams:
                l.append(" ".join(j))
    
        return l

    def replace_consecutive(self, sentence):
        sentence = re.sub(r"(.)\1+", r"\1\1", sentence)
        return sentence

    def extract_emojis(self, sentence):
        plain_text = []
        emo = []
        for c in sentence:
            if c not in emoji.UNICODE_EMOJI:
                plain_text.append(c)
            else:
                emo.append(c)
        plain_text = "".join(plain_text)
        return plain_text, emo

    def remove_punctuations(self, sentence):
        result = "".join([w if w not in self.punctuations and not w.isdigit() else "" for w in sentence])
        return result

    def word_lowercase(self, sentence):
        return sentence.lower()

    def word_stemmer(self, sentence):
        sentence = self.stemmer.stem(sentence)
        return sentence

    def clean(self, tokens, types):
        results = []
        for tok in tokens:
            if "remove_emojis" in types:
                pass
            else:
                tok, emo = self.extract_emojis(tok)  
            if "base" in types or "lower" in types:
                tok = self.word_lowercase(tok)
            
            if "base" in types or "remove_punctuaions" in types:
                tok = self.remove_punctuations(tok)
                if tok == '':
                        continue

            if "base" in types or "remove_stopwords" in types:
                tok = self.remove_stopwords(tok)
                if tok == '':
                        continue
            
            if "stem" in types:
                tok = self.word_stemmer(tok)
            
            if "replace_consecutive" in types:
                tok = self.replace_consecutive(tok)
            
            if (tok is not None) and (not tok.isspace()) and (tok!= ''): 
                results.append(tok)  
            if "remove_emojis" in types:
                pass
            else:
                results += emo
                
        grams = self.add_n_grams(results) if "ngrams" in types else []
        results = results + grams
        
        if len(results) == 0:
            results.append('NaN')
        return results
    
       