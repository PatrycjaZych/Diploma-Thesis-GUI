import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import seaborn as sns

# NLP tools initialization
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()


# Function to create a custom set of stop words
def create_stop_words():
    stop_words = """
    a about across after afterwards again all almost alone along
    already also although always am among amongst amount an and another any anyhow
    anyone anything anyway anywhere are as at

    be became because become becomes becoming been before beforehand behind
    being beside besides between both but by

    can cannot ca could

    did do does doing done due during

    each either else elsewhere empty enough even ever every
    everyone everything everywhere except

    few for former formerly from full
    further

    give

    had has have he hence her here hereafter hereby herein hereupon hers herself
    him himself his how however

    i if in indeed into is it its itself

    keep

    last latter latterly least less

    just

    made make many may me meanwhile might mine more moreover most mostly much
    must my myself

    name namely neither never nevertheless next no nobody none noone nor not
    nothing now nowhere

    of often on once only onto or other others otherwise our ours ourselves
    out own

    part per perhaps please put

    quite

    rather re really regarding

    same say see seem seemed seeming seems serious several she should show side
    since so some somehow someone something sometime sometimes somewhere still such

    take than that the their them themselves then thence there thereafter
    thereby therefore therein thereupon these they third this those though through
    throughout thus to together too toward towards

    under until unless upon us used using

    various very via was we well were what whatever when whence whenever where
    whereafter whereas whereby wherein whereupon wherever whether which while
    whither who whoever whole whom whose why will with within without would

    yet you your yours yourself yourselves
    """
    STOP_WORDS = set(stop_words.split())

    # Adding contractions
    contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
    STOP_WORDS.update(contractions)

    # Adding typographical apostrophe variants
    for apostrophe in ["‘", "’"]:
        for stopword in contractions:
            STOP_WORDS.add(stopword.replace("'", apostrophe))

    return STOP_WORDS


# Initializing the custom stop words set
stop_words_set = create_stop_words()


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^-a-zA-Z0-9+%\s]", "", text)  # Remove unnecessary characters
    doc = nlp(text)  # Tokenize using SpaCy
    lemmatized_text = " ".join(
        [token.lemma_ for token in doc if token.lemma_ not in stop_words_set])  # Lemmatize and remove stop words
    return lemmatized_text.strip()


# Sentiment analysis function using custom financial sentiment rules
def calculate_sentiment_vader_custom(raw_news, preprocessed_news):
    sentiment_score = sia.polarity_scores(raw_news)['compound']

    # Positive words specific to financial news
    positive_words = {
        "upturn", "bullish", "rally", "advance", "expansion", "breakthrough",
        "record high", "lucrative", "prosperity", "fortune", "thrive", "inflow",
        "rebound", "beat", "strategic alliance", "upbeat outlook",
        "milestone", "partnership", "share buyback", "dividend raise", "ipo success",
        "profit surge", "rise", "soar", "bull", "raise", "generate", "noteworthy",
        "surge", "radar", "phenomenal", "earn", "trend stock"
    }

    # Negative words specific to financial news
    negative_words = {
        "decline", "fall", "bearish", "plunge", "slump", "downward", "concern",
        "downturn", "outflow", "stagnation", "layoff", "bankruptcy", "underperform",
        "volatility", "selloff", "sell-off", "downgrade", "recession fear", "shortfall",
        "plummet", "bear market", "drop in value", "bankruptcy proceeding",
        "cut forecast", "miss estimate", "downward pressure", "production cut",
        "regulatory setback", "settlement charge", "supply chain disruption", "bear",
        "stock down", "step down", "stock falter"
    }

    # Adjust sentiment score based on financial context
    sentiment_score += 0.1 * sum(word in preprocessed_news for word in positive_words)
    sentiment_score -= 0.2 * sum(word in preprocessed_news for word in negative_words)

    # Contextual checks for percentage changes
    if any(word in preprocessed_news for word in
           ["up", "surge", "rise", "add", "soar", "jump", "climb", "rocket", "race ahead", "yield over",
            "move"]) and '%' in preprocessed_news:
        sentiment_score += 0.3
    if any(word in preprocessed_news for word in
           ["down", "fall", "decline", "decrease", "plunge", "drop", "dip", "-", "slide"]) and '%' in preprocessed_news:
        sentiment_score -= 0.4

    return max(min(sentiment_score, 1), -1)

# GUI starts here

# Dodanie stylu CSS do wyrównania tabeli do środka
st.markdown(
    """
    <style>
    .main .block-container {
        padding: 1rem;
        max-width: 100% !important;
    }
    .upload-col {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tytuł aplikacji
st.title("Sentiment detection and evaluation of the impact of external information on the price of selected NASDAQ-listed companies")


st.info("""
This application is part of a diploma thesis titled "Sentiment Detection and Evaluation of the Impact of External Information on
the Price of Selected NASDAQ-Listed Companies". The tool is designed to perform sentiment analysis on news articles related to 
specific stocks, and to evaluate how external information, such as financial news, influences stock prices. By uploading a CSV 
file containing stock price data and a text file with news headers, the user can explore the correlation between sentiment 
and stock prices. The application provides a user-friendly interface to analyze these relationships and gain insights into the 
impact of news sentiment on stock performance.
""")


st.header("Upload Files")


st.info("""
Please upload the stock prices in CSV format and sentiment data in text format.
The stock prices file should contain historical data, including dates and closing prices.
The sentiment data should contain headers of news articles related to the stock with dates.
""")


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Stock Prices CSV File")
    uploaded_stock_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_stock_file is not None:
        st.info("CSV File Preview")
        # Wczytaj dane z pliku CSV
        df_stock = pd.read_csv(uploaded_stock_file)
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])  # Konwersja kolumny na datetime
        #df_stock["Date"] = df_stock["Date"].dt.strftime("%Y-%m-%d")
        st.write("Stock prices data:")
        #st.write(df_stock.head())
        st.dataframe(df_stock, height=400)

with col2:
    st.subheader("Upload Sentiment Data Text File")
    uploaded_sentiment_file = st.file_uploader("Choose a text file", type="txt")
    if uploaded_sentiment_file is not None:
        # Wczytaj dane z pliku tekstowego
        df_sentiment = pd.read_csv(uploaded_sentiment_file, delimiter=';', usecols=['news_header', 'news_date'])

        df_sentiment['news_date'] = pd.to_datetime(df_sentiment['news_date'])  # Konwersja kolumny na datetime
        #df_sentiment["news_date"] = df_sentiment["news_date"].dt.strftime("%Y-%m-%d")
        # Przetwarzanie nagłówków
        st.info("Preprocessing the headlines and calculating sentiment scores using VADER.")
        df_sentiment['news_header_preprocessed'] = df_sentiment['news_header'].apply(preprocess_text)
      #  df_sentiment['sentiment_vader'] = df_sentiment['news_header_preprocessed'].apply(calculate_sentiment_vader)

        df_sentiment['sentiment_vader'] = df_sentiment.apply(
            lambda row: calculate_sentiment_vader_custom(row['news_header'], row['news_header_preprocessed']), axis=1)

        # Grupowanie danych na podstawie oryginalnych nagłówków (bez sentymentu)
        df_grouped = df_sentiment.groupby('news_date').agg({
            'news_header': lambda x: ' ; '.join(x)  # Użycie nieprzetworzonych nagłówków
        }).reset_index()

        # Wyświetlenie grupowanych danych bez sentymentu
        st.write("Grouped Data with Headers:")
        st.dataframe(df_grouped, height=400)

        # Wyświetlenie przetworzonych danych z sentymentem
        st.write("Preprocessed Data with Calculated Sentiment:")
        st.dataframe(df_sentiment[['news_date', 'news_header_preprocessed', 'sentiment_vader']], height=400)
        st.info("The headlines from the text file underwent preliminary processing. Using appropriate functions, the text was converted to lowercase, unnecessary characters were removed, lemmatization was applied, and stop words were excluded.")
        # Create a DataFrame with original headers, preprocessed text, and sentiment scores
        df_export = df_sentiment[['news_date', 'news_header', 'news_header_preprocessed', 'sentiment_vader']]

        # Exporting the full DataFrame to CSV
        df_export.to_csv("SentimentResults.csv", sep='|', index=False)
        st.success("Analysis complete. Results saved to SentimentResults.csv")

        # Creating the CSV download button
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sentiment Results as CSV",
            data=csv,
            file_name='SentimentResults.csv',
            mime='text/csv',
        )

# Przycisk do uruchomienia analizy
if uploaded_stock_file is not None and uploaded_sentiment_file is not None:
    st.header("Analysis Results")

    # Przetwarzanie danych
    df_stock['Return'] = df_stock['Close'].pct_change() * 100
    df_stock['Return'] = df_stock['Return'].shift(-1)

    df_sentiment = df_sentiment[['news_date', 'sentiment_vader']]
    df_merged = pd.merge(df_sentiment, df_stock, left_on='news_date', right_on='Date', how='left')
    df_merged = df_merged[['news_date', 'sentiment_vader', 'Return']]
    df_merged = df_merged.dropna(subset=['Return'])
    df_merged = df_merged.reset_index(drop=True)

    df_filtered = df_merged.query('sentiment_vader != 0.00000')
    df_filtered = df_filtered.reset_index(drop=True)

    # Now adding histogram and pie chart analysis

    # Merge sentiment data with stock data
    merged_df = pd.merge(df_stock, df_sentiment[['news_date', 'sentiment_vader']], left_on='Date',
                         right_on='news_date', how='inner')

    # Możesz teraz obliczyć 'Arithmetic_Return' i inne kolumny, jak wcześniej
    merged_df['Arithmetic_Return'] = ((merged_df['Open'].shift(-1) - merged_df['Open']) / merged_df['Open']) * 100


    # Dodanie kolumny 'alignment0.1', jak w poprzednim kodzie
    def categorize_alignment(row, cutoff):
        sentiment = row['sentiment_vader']
        arith_return = row['Arithmetic_Return']

        if (-cutoff <= arith_return <= cutoff):
            return 'Small ror - no influence'
        elif (sentiment > 0 and arith_return > cutoff) or (sentiment < 0 and arith_return < -cutoff):
            return 'Aligned'
        else:
            return 'Not Aligned'

    # Apply the categorization function
    merged_df['alignment0.1'] = merged_df.apply(categorize_alignment, axis=1, cutoff=0.0)

    # Sekcja kolumn
    col1, col2 = st.columns([1, 1])  # Kolumny o tej samej szerokości 50% na ekranie

    # Wykres w kolumnie 2
    with col1:
        st.subheader("Stock Open Price with Sentiment Alignment. ")

        # Przygotowanie wykresu
        plt.figure(figsize=(12, 6))

        # Wykres ceny otwarcia
        plt.plot(merged_df['Date'], merged_df['Open'], color='black', label='Open Price', alpha=0.6)

        # Dodanie wskaźników zgodności
        for index, row in merged_df.iterrows():
            if row['alignment0.1'] == 'Aligned':
                plt.text(row['Date'], row['Open'] + 0.5, '✔', color='green', fontsize=12,
                         ha='center')  # Zielony check mark
            elif row['alignment0.1'] == 'Not Aligned':
                plt.text(row['Date'], row['Open'] - 0.5, '✘', color='red', fontsize=12, ha='center')  # Czerwony krzyż
        #    elif row['alignment0.1'] == 'Small ror - no influence':
        #        plt.text(row['Date'], row['Open'], '○', color='blue', fontsize=12, ha='center')  # Niebieskie kółko

        # Dodanie tytułu, etykiet i legendy
        plt.title('Stock Open Price Trend with Sentiment Alignment')
        plt.xlabel('Date')
        plt.ylabel('Open Price')
        plt.legend(loc='best')
        plt.grid(True)

        # Wyświetlenie wykresu
        st.pyplot(plt)
    with col2:
        st.info("Stock Open Price with Sentiment Alignment. Influence - represented by a green check mark, it indicates that the sentiment and the rate of return for the given day were in the same direction, Reverse Influence - represented by a red cross, it indicates that the sentiment and the rate of return for the given day were opposite. ")
    st.subheader("Histogram and Pie Chart")

    # Tworzymy kolumny, aby wykresy były obok siebie w jednej linii
    col1, col2 = st.columns([1, 1])  # Możesz dostosować proporcje kolumn jeśli chcesz

    # HISTOGRAM: Distribution of Sentiment Scores
    with col1:
        st.info("This histogram shows distribution of sentiment scores. The histogram above displays the distribution of sentiment scores, broken into 32 bins. The scale ranges from -1.0 to 1.0. Each bin represents a segment of the sentiment spectrum. X-axis represents the sentiment scores calculated using the VADER sentiment analysis tool. Y-axis shows the number of news headlines corresponding to each bin.")

        plt.figure(figsize=(6, 4))  # Zmniejszamy rozmiar wykresu
        sns.histplot(df_sentiment['sentiment_vader'], bins=32, kde=True, color='blue')
        plt.title('Distribution of Sentiment Scores', fontsize=16)
        plt.xlabel('Sentiment Score (VADER)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True)
        st.pyplot(plt)


    # PIE CHART: Sentiment Classification
    with col2:
        st.info("This pie chart classifies sentiment into Positive, Neutral, and Negative categories. The chart provides a clear visual representation of the spread of sentiment within the dataset.Headlines were categorized as follows: Positive: Sentiment score > 0 (green) Neutral: Sentiment score = 0 (blue) Negative: Sentiment score < 0 (red) ")

        # Sentiment classification into categories
        sentiment_counts = df_sentiment['sentiment_vader'].apply(
            lambda x: 'Positive' if x > 0 else ('Neutral' if x == 0 else 'Negative')
        ).value_counts()

        # Define colors for the pie chart
        color_mapping = {
            'Positive': '#8db670',  # Green for positive
            'Neutral': '#d4edf8',  # Blue for neutral
            'Negative': '#a32b32'  # Red for negative
        }
        colors = [color_mapping[label] for label in sentiment_counts.index]

        plt.figure(figsize=(6, 6))  # Zmniejszamy rozmiar wykresu
        plt.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100. * sum(sentiment_counts)))})",
            startangle=140,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        plt.title('Sentiment Classification', fontsize=16, fontweight='bold')
        st.pyplot(plt)

        # Sekcja ze słówkami kluczowymi
st.header("Key Terms")

# Definicje słów kluczowych
key_terms = {
    "Sentiment": "An attitude, thought, or judgment prompted by feeling, whereas opinion is defined as a view, judgment, or belief formed about something, often based on reasoning, experience, or understanding, rather than emotional response.",
    "Sentiment Analysis": "Sentiment analysis, also called opinion mining, is the field of study that analyzes people’s opinions, sentiments, appraisals, attitudes, and emotions toward entities and their attributes expressed in written text.",
    "Stock Prices": "The cost of purchasing a share of a company’s stock. It is determined by the supply and demand in the stock market.",
    "Lemmanization": "Process that ensures the output word is an existing normalized form of the word (for example, lemma) that can be found in the dictionary.",
    "Stop words": "Common words that carry little (or perhaps no) meaningful information.",
    "VADER": "simple rule-based model for general sentiment analysis.",
}

# Wybór słówka kluczowego z listy
term = st.selectbox("Select a key term to get its definition:", list(key_terms.keys()))

# Wyświetlenie definicji
if term:
    st.write(f"**Definition of {term}:** {key_terms[term]}")

# Sekcja z informacjami o autorach, promotorze i recenzencie
st.header("Diploma Thesis Information")

# Użycie układu kolumn dla informacji o tytule, autorach, promotorze i recenzencie
info_col1, info_col2, info_col3 = st.columns([1, 2, 1])

with info_col1:
    st.write("**Topic:**")
    st.write("Sentiment Detection and Evaluation of the Impact of External Information on the Price of Selected NASDAQ-Listed Companies")

with info_col2:
    st.write("**Authors:**")
    st.markdown(
        """
        <div class="authors-col">
            <p> Alicja Tomaszewska</p>
            <p> Julita Bussler</p>
            <p> Patrycja Zych</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with info_col3:
    st.write("**Supervisor:**")
    st.write("Dr Paweł Weichbroth")
    st.write("**Reviewer:**")
    st.write("Dr inż. Anna Trzaskowska")