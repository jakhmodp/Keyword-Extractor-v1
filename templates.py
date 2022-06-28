HTML_BANNER = """
<div style="background-color:royalblue;padding:10px;">
<h1 style="color:white;text-align:center;">Keyword Extractor for NLP </h1> </div>

"""

SPACY = """ It works as follows:
* Split the input text content by tokens
* Extract the hot words from the token list.
* Set the hot words as the words with pos tag “PROPN“, “ADJ“, or “NOUN“. (POS tag list is customizable)
* Find the most common T number of hot words from the list
* Print the results

To learn more, click [https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp/]
"""

KEYBERT = """
**Keybert** is a basic and easy-to-use keyword extraction technique that generates the most similar keywords and keyphrases to a given document using BERT embeddings. It uses BERT-embeddings and basic cosine similarity to locate the sub-documents in a document that are the most similar to the document itself.

BERT is used to extract document embeddings in order to obtain a document-level representation. The word embeddings for N-gram words/phrases are then extracted. Finally, it uses cosine similarity to find the words/phrases that are most similar to the document. The most comparable terms can then be identified as the ones that best describe the entire document.

Because it is built on BERT, KeyBert generates embeddings using huggingface transformer-based pre-trained models. The all-MiniLM-L6-v2 model is used by default for embedding.

To learn more, click [https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp/]

"""
TFIDF = """
**TF-IDF** (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents.
This is done by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents.
To learn more, click [https://monkeylearn.com/blog/what-is-tf-idf/]
"""
WORDCOUNT = """
Word count is the simplest way of finding the important words by checking the frequency of the word i.e. number of times it appears in the document.

"""
YAKE="""
In **Yet Another Keyword Extractor (Yake)**, for automatic keyword extraction, text features are exploited in an unsupervised manner. YAKE is a basic unsupervised automatic keyword extraction method that identifies the most relevant keywords in a text by using text statistical data from single texts. This technique does not rely on dictionaries, external corpora, text size, language, or domain, and it does not require training on a specific set of documents. The Yake algorithm’s major characteristics are as follows:

* Unsupervised approach
* Corpus-Independent
* Domain and Language Independent
* Single-Document
"""

RAKE="""
**RAKE (Rapid Automatic Keyword Extraction)** is a well-known keyword extraction method that finds the most relevant words or phrases in a piece of text using a set of stopwords and phrase delimiters. Rake nltk is an expanded version of RAKE that is supported by NLTK. The steps for Rapid Automatic Keyword Extraction are as follows:

* Split the input text content by dotes
* Create a matrix of word co-occurrences
* Word scoring – That score can be calculated as the degree of a word in the matrix, as the word frequency, or as the degree of the word divided by its frequency
* keyphrases can also create by combining the keywords
* A keyword or keyphrase is chosen if and only if its score belongs to the top T scores where T is the number of keywords you want to extract

To learn more, click [https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp/]

"""

TEXTRANK = """
**Textrank** is a Python tool that extracts keywords and summarises text. The algorithm determines how closely words are related by looking at whether they follow one another. The most important terms in the text are then ranked using the PageRank algorithm. Textrank is usually compatible with the Spacy pipeline. Here are the primary processes Textrank does while extracting keywords from a document.

**Step – 1:** In order to find relevant terms, the Textrank algorithm creates a word network (word graph). This network is created by looking at which words are linked to one another. If two words appear frequently next to each other in the text, a link is established between them. The link is given more weight if the two words appear more frequently next to each other.

**Step – 2:** To identify the relevance of each word, the Pagerank algorithm is applied to the formed network. The top third of each of these terms is kept and considered important. Then, if relevant terms appear in the text after one another, a keywords table is constructed by grouping them together.

TextRank is a Python implementation that allows for fast and accurate phrase extraction as well as extractive summarization for use in spaCy workflows. The graph method isn’t reliant on any specific natural language and doesn’t require domain knowledge

To learn more, click [https://www.analyticsvidhya.com/blog/2022/03/keyword-extraction-methods-from-documents-in-nlp/]

"""
