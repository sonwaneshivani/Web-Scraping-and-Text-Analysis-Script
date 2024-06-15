## Web Scraping and Text Analysis Script
This Python script performs web scraping, text processing, and analysis on a dataset of URLs. It generates an output file containing various metrics and analysis results.

### Dependencies
Python 3.x
pandas
requests
BeautifulSoup
regex
nltk
You can install the required dependencies using pip:
pip install pandas requests beautifulsoup4 regex nltk

### How to Run
Clone the Repository: Clone this repository to your local machine.

### Install Dependencies: 
Install the required dependencies using the command mentioned above.

### Prepare Dataset: 
Ensure that you have a dataset named Input.xlsx in the Dataset directory. This dataset should contain two columns: URL and URL_ID.

### Run the Script: 
Execute the Python script web_scraping_text_analysis.py using the following command:
python web_scraping_text_analysis.py

Check Output: Once the script execution is complete, you'll find the output file named Output Data Structure.xlsx in the root directory. This file contains the analysis results.

### Approach
Web Scraping: The script iterates through the URLs provided in the dataset, scrapes the text content from the web pages, and stores it in a DataFrame.

Text Processing: The scraped text is preprocessed by converting it to lowercase, removing non-alphabetic characters, tokenizing, removing stopwords, and lemmatizing.

Sentiment Analysis: Positive and negative word counts are calculated based on predefined dictionaries of positive and negative words. Sentiment score and polar score are calculated.

Text Metrics Calculation: Various text metrics such as average sentence length, percentage of complex words, Fog index, average words per sentence, etc., are calculated using NLTK functions.

Output Generation: The calculated metrics are added to the DataFrame, unnecessary columns are dropped, and the final DataFrame is saved to an Excel file.
#
