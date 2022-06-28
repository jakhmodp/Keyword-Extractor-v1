import streamlit as st
import streamlit.components.v1 as stc
import pandas as pd
import templates
import about

import nltk
nltk.download('punkt')

import keywordExtractor_functions as KeyFunc
from nltk import sent_tokenize, word_tokenize
import re
from neattext import functions as fxn

# File processing packages
import docx2txt
#import pyPDF2

# Data Visualization Packages
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud

# NLP packages
from collections import Counter

def textclean(text):
    ct = text.lower()
    ct = fxn.remove_emails(ct)
    ct = fxn.remove_html_tags(ct)
    ct = fxn.remove_urls(ct)
    ct = re.sub('[^a-zA-Z\.]', ' ', ct)
    ct = fxn.remove_multiple_spaces(ct)
    ct = fxn.remove_stopwords(ct)
    ct = fxn.remove_userhandles(ct)
    ct = re.sub(r'\.\s\.', '.', ct)
    ct = re.sub('\.+', '.', ct)
    ct = sent_tokenize(ct)
    ct = ' '.join([sent for sent in ct if len(sent) > 2])
    #clean_text = [word for word in raw_text.split(" ") if word not in stopwords.words('english')]
    return ct



# Converting dictionary to DataFrame
def dict2dataFrame(dict, columnNames):
    dataset = pd.DataFrame(dict.items())
    dataset.columns = columnNames
    return dataset


def draw_dataframe(dataframe):
    st.dataframe(dataframe)


def draw_chart(dataFrame):
    my_chart = alt.Chart(dataFrame).mark_bar()\
        .encode(
                alt.X('Feature', sort=alt.SortField('Score', order='descending')),
                alt.Y('Score')
                )#.properties(width=700)
    st.altair_chart(my_chart, use_container_width=True)


def draw_wordcloud(dataFrame):
    fig, ax = plt.subplots()
    text_for_wordCloud = " ".join(dataFrame['Feature'].tolist())
    wordcloud = WordCloud(background_color ='white').generate(text_for_wordCloud)
    #fig.update_layout(height=800)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)


def text_highlighter(text, keywords):
    word_tokens = word_tokenize(text)
    for i, word in enumerate(word_tokens):
        if word.lower() in keywords:
            word_tokens[i] = '<mark>' + word + '</mark>'

    return ' '.join(word_tokens)


def show_content(kw_method, raw_text):
    kw_top_n = st.slider("Show top 'n' results", min_value=0, max_value=100, value=20, step=5)
    # if st.button("Extract"):
    st.info(f"Top Keywords in the provided text are extracted using:: {kw_method}")

    if kw_method == 'TextRank':
        if len(raw_text) > 0:
            clean_text = textclean(raw_text)
            dataset = KeyFunc.extract_textrank(clean_text, kw_top_n)

    elif kw_method == 'TFIDF':
        if len(raw_text) > 0:
            clean_text = textclean(raw_text)
            list_of_sents = sent_tokenize(clean_text)
            dataset = KeyFunc.extract_TFIDF(list_of_sents, kw_top_n)


    elif kw_method == 'Spacy POS-based':
        if len(raw_text) > 0:
            clean_text = textclean(raw_text)
            dataset = KeyFunc.extract_spacy(clean_text, kw_top_n)


    elif kw_method == 'KeyBERT':
        if len(raw_text) > 0:
            clean_text = textclean(raw_text)
            dataset = KeyFunc.extract_keybert(clean_text, kw_top_n)


    elif kw_method == 'Yake':
        if len(raw_text) > 0:
            clean_text = textclean(raw_text)
            list_of_sents = sent_tokenize(clean_text)
            dataset = KeyFunc.extract_TFIDF(list_of_sents, kw_top_n)


    elif kw_method == 'Rake':
        if len(raw_text) > 0:
            clean_text = textclean(raw_text)
            list_of_sents = sent_tokenize(clean_text)
            dataset = KeyFunc.extract_TFIDF(list_of_sents, kw_top_n)


    else:
        if len(raw_text) > 0:
            clean_text = textclean(raw_text)
            word_list = [word for word in word_tokenize(clean_text) if len(word) > 2]
            keywords_in_text = Counter(word_list)
            # st.write(keywords_in_text)
            # Printing DataFrame in the screen
            top_n_words = keywords_in_text.most_common(kw_top_n)
            dictionary_of_keywords = {pair[0]: pair[1] for pair in top_n_words}
            dataset = dict2dataFrame(dictionary_of_keywords, ['Feature', 'Score'])
            dataset.sort_values('Score', ascending=False, inplace=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
                    #### Top {kw_top_n} words based on the importance
                    """)

        draw_chart(dataset)

    with col2:
        st.markdown(f"""
                                #### Top {kw_top_n} words based on the frequency
                                """)
        draw_wordcloud(dataset)

    exp = st.expander("Output Text")

    with exp:
        text = text_highlighter(raw_text, dataset['Feature'].tolist())
        htmltext = f'<p>{text}</p>'  #
        stc.html(htmltext, height=250, scrolling=True)

def main():
    st.set_page_config(layout="wide")
    st.sidebar.image(
        "https://res.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/z3ahdkytzwi1jxlpazje",
        width=50,
    )
    image = Image.open('banner.JPG')
    st.image(image)

    #stc.html(templates.HTML_BANNER)
    menu = ['Home', 'Dropfiles', 'About']
    choice = st.sidebar.selectbox("Menu", menu)


    if choice == 'Home':
        keyword_extraction_methods = ['WordCount',
                                      'TextRank',
                                      'TFIDF',
                                      'Yake',
                                      'Rake',
                                      'Keybert',
                                      'Spacy POS-based'
                                      ]
        kw_method = st.sidebar.selectbox("Methods", keyword_extraction_methods)

        # print information about the applied metho
        method_info = st.sidebar.expander(f'about {kw_method}')
        with method_info:
            if kw_method == 'WordCount':
                st.markdown(templates.WORDCOUNT)
            elif kw_method == 'TextRank':
                st.markdown(templates.TEXTRANK)
            elif kw_method == 'TFIDF':
                st.markdown(templates.TFIDF)
            elif kw_method == 'Yake':
                st.markdown(templates.YAKE)
            elif kw_method == 'Rake':
                st.markdown(templates.RAKE)
            elif kw_method == 'Keybert':
                st.markdown(templates.KEYBERT)
            else:
                st.markdown(templates.SPACY)


        st.markdown("""### Home""")
        st.write("""Extract keywords from an input text. Choose from a wide variety of functions 
        and visualization. You can also choose the number of values in the result.""")
        raw_text = st.text_area("Enter text here")
        if len(raw_text) <= 0:
            st.error("Text must be entered")
        else:
            show_content(kw_method, raw_text)



    elif choice == 'Dropfiles':
        st.subheader("Dropfiles")
        raw_text_file = st.file_uploader("Upload file here (txt,  docx)", type=['txt', 'docx'])
        keyword_extraction_methods = ['WordCount',
                                      'TextRank',
                                      'TFIDF',
                                      'Yake',
                                      'Rake',
                                      'Keybert',
                                      'Spacy POS-based'
                                      ]
        kw_method = st.sidebar.selectbox("Methods", keyword_extraction_methods)

        # print information about the applied metho
        method_info = st.sidebar.expander(f'about {kw_method}')
        with method_info:
            if kw_method == 'WordCount':
                st.markdown(templates.WORDCOUNT)
            elif kw_method == 'TextRank':
                st.markdown(templates.TEXTRANK)
            elif kw_method == 'TFIDF':
                st.markdown(templates.TFIDF)
            elif kw_method == 'Yake':
                st.markdown(templates.YAKE)
            elif kw_method == 'Rake':
                st.markdown(templates.RAKE)
            elif kw_method == 'Keybert':
                st.markdown(templates.KEYBERT)
            else:
                st.markdown(templates.SPACY)

        if st.button("Extract"):
            if raw_text_file is None:
                st.error("No file provided")
            else:
                if raw_text_file.type not in ['text/plain', 'application/octet-stream']:
                    st.info(f"Invalid file type! {raw_text_file.type}")
                else:
                    if raw_text_file.type == 'application/octet-stream':
                        raw_text = docx2txt.process(raw_text_file)
                        show_content(kw_method, raw_text)
                    elif raw_text_file.type == 'text/plain':
                        raw_text = raw_text_file.getvalue().decode("utf-8")
                        show_content(kw_method, raw_text)
                        #raw_text = raw_text_file.read()
                        #show_content(kw_method, raw_text)


    else:
        about_expander = st.expander("About Keyword Extractor",expanded=True )
        with about_expander:
            st.markdown(about.about)

        st.subheader("Structure of the application")

        input_expander = st.expander("Input", expanded=True)
        with input_expander:
            st.markdown(about.input)
        output_expander = st.expander("Output", expanded=True)
        with output_expander:
            st.markdown(about.output)







if __name__ == '__main__':
    main()


