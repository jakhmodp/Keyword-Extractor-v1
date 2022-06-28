about = """

**Keyword extraction** is the automated process of extracting the most relevant words and expressions from text.

This application is showing how the Keywords are being extracted using various tools:

* Spacy
* Keybert
* TfIdf
* Wordcloud
* Yake
* Rake
* Textrank
"""


input = """
When user opens this application, a home page is open as default. In the center screen, a text box is provided \
where user has to enter the text. A default method to extract the keywords is set to **word count** which is \
the simplest of all the methods. After entering the text when the user clicks anywhere in the screen, application \
starts processing the text. 

User can also choose any other method available in the left side bar under **methods** menu.

After the user has run the program for first time, he will get an option to change the number of keywords between \
0 to 100. **Default value** is **20**.

Another way to provide the input is by uploaded the data files. Presently, we have provided only two options \
- text files (.txt) and word documents (.docx). \
For this, choose **Dropfiles** option from **Menu** on the side bar.
"""

output = """

Application generates two type of graphs - a **bar plot** which shows features which are the most important words based on the \
score generated by the selected algorithm/method.
Another graph is the **word cloud** which shows the important words based on the word frequencies.

In the bottom of the screen, a section 'output text' is provided which shows highlights the keywords directly in the text \
document.

"""