# Keyword-Extractor-v1
Python NLP application to extract the keywords from a text input or file.

**Keyword extraction** is the automated process of extracting the most relevant words and expressions from text.

This application is showing how the Keywords are being extracted using various tools:

- Spacy
- Keybert
- TfIdf
- Wordcloud
- Yake
- Rake
- Textrank

## Structure of the application
### Input:
When user opens this application, a home page is open as default. In the center screen, a text box is provided where user has to enter the text. A default method to extract the keywords is set to **word count** which is the simplest of all the methods. After entering the text when the user clicks anywhere in the screen, application starts processing the text. 

User can also choose any other method available in the left side bar under **methods** menu.

After the user has run the program for first time, he will get an option to change the number of keywords between 0 to 100. **Default value** is **20**.

Another way to provide the input is by uploaded the data files. Presently, we have provided only two options 1) text files (.txt) and word documents (.docx). For this, choose **Dropfiles** option from **Menu** on the side bar.

### Output:

Application generates two type of graphs - a **bar plot** which shows features which are the most important words based on the score generated by the selected algorithm/method.
Another graph is the **word cloud** which shows the important words based on the word frequencies.

In the bottom of the screen, a section 'output text' is provided which shows highlights the keywords directly in the text document.

![image](https://user-images.githubusercontent.com/93132125/176247675-ad2e09c6-7927-4d40-8fb2-07fbf82f6c0b.png)
![image](https://user-images.githubusercontent.com/93132125/176247773-1ef47ff2-0ac8-4d92-b073-fccd08d1d577.png)
![image](https://user-images.githubusercontent.com/93132125/176247829-9f92465e-3760-4b01-a1ec-8cc41f7726b4.png)
![image](https://user-images.githubusercontent.com/93132125/176247941-0b66b85a-d4e9-445d-aaa5-919fb5951a17.png)
![image](https://user-images.githubusercontent.com/93132125/176248134-62f4224e-6135-437a-b39f-2f138edf95aa.png)


# Python Environment
python-3.8.13
