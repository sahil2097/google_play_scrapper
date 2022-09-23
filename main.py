import pandas as pd
import re
import os
import logging
import nltk
# from PIL import Image
import pymongo
import json
from nltk.corpus import stopwords
from autocorrect import Speller
from nltk.stem.wordnet import WordNetLemmatizer
# from nltk import word_tokenize
from nltk.tag import pos_tag
# from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from google_play_scraper import app, search, reviews, Sort
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin

# import csvkit

logging.basicConfig(filename='google_play.log', level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')

IMG_FOLDER = os.path.join('static', 'images')

appp = Flask(__name__)

# config environment variables
appp.config['IMG_FOLDER'] = IMG_FOLDER


@appp.route('/', methods=['POST','GET'])  # Display the home page
@cross_origin()
def homepage():
    return render_template("index.html")

@appp.route('/debug')
def check():
    return jsonify( appp.config['IMG_FOLDER'])

@appp.route('/review', methods=['GET', 'POST'])
@cross_origin()
def fetch_review():
    """This function is used to do analysis of apps description and reviews of apps on google play store """
    if request.method == 'POST':
        try:
            operation = request.form['operation']
            searchstring = request.form['content']
            no_of_apps = int(request.form['total_apps'])
            result = search(searchstring,
                            lang="en",  # defaults to 'en'
                            country='in',  # defaults to 'us'
                            n_hits=no_of_apps  # defaults to 30 (= Google's maximum)
                            )
            result_df = pd.DataFrame(result)
            app_ids = list(result_df['appId'])  # getting the list app id's from related keyword search
            if (operation) == 'description_analysis':  # Description analysis from description on google play store
                try:
                    logging.info('Description analysis started')
                    result_details = []
                    for i in app_ids:
                        result_app = app(str(i)) # getting app details of app id's captured
                        del result_app['comments']
                        result_details.append(result_app)
                    app_details = pd.DataFrame(result_details)
                    logging.info('Apps details captured')
                    try:
                        app_details.to_csv(searchstring+'_data.csv')
                        logging.info('Data Downloaded in CSV format')
                    except Exception as e:
                        logging.exception(e)

                    df_desc_list = []
                    for i in app_details['description']:
                        df_desc_list.append(i)
                    description_string = re.sub("[^A-Za-z" "]+", " ", str(df_desc_list)).lower()
                    description_spelling_correction = []
                    spell = Speller(lang='en')

                    for i in description_string.split():  # spell check
                        description_spelling_correction.append(spell(i))
                    logging.info(" Spell check done")

                    lemmatizer = WordNetLemmatizer()
                    description_lemmas = []
                    for i in description_spelling_correction:  # converting to base words
                        description_lemmas.append(lemmatizer.lemmatize(i))

                    stop_words = set(stopwords.words('english'))
                    description_filtered_sentence = []
                    for w in description_lemmas:  # stop words removal
                        if w not in stop_words:
                            description_filtered_sentence.append(w)
                    desc_bigrams_list = list(nltk.bigrams(description_filtered_sentence))
                    desc_dictionary2 = [' '.join(tup) for tup in desc_bigrams_list]
                    from sklearn.feature_extraction.text import CountVectorizer
                    vectorizer = CountVectorizer(ngram_range=(2, 2))
                    desc_bag_of_words = vectorizer.fit_transform(desc_dictionary2)
                    # vectorizer.vocabulary_

                    sum_words_desc = desc_bag_of_words.sum(axis=0)
                    words_freq_desc = [(word, sum_words_desc[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
                    words_freq_desc = sorted(words_freq_desc, key=lambda x: x[1], reverse=True)
                    words_dict_desc = dict(words_freq_desc)
                    WC_height = 1000
                    WC_width = 1500
                    WC_max_words = 100
                    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)

                    wordCloud.generate_from_frequencies(words_dict_desc)
                    plt.figure(4)
                    plt.title('Most frequently occurring bigrams connected by same colour and font size')
                    plt.imshow(wordCloud, interpolation='bilinear')
                    plt.axis("off")
                    # create path to save wc image
                    image_path = os.path.join(appp.config['IMG_FOLDER'], searchstring +'_description'+ '.png')
                    plt.savefig(image_path)
                    #img_file = os.listdir(appp.config['IMG_FOLDER'])[0]

                    #full_filename = os.path.join(appp.config['IMG_FOLDER'], img_file)
                    full_filename = image_path
                    logging.info(' word cloud created successfully')
                except Exception as e:
                    logging.info('Error in description Analysis')
                    logging.exception(e)

            if (operation) == 'review_analysis':
                try:
                    logging.info('Review Analysis started')
                    reviews_full = []
                    for i in app_ids:
                        reviews_1, _ = reviews(str(i), count=200)  # Getting 200 reviews details of each app id
                        reviews_full.append(reviews_1)
                    reviews_full_df = pd.DataFrame(reviews_full)
                    logging.info(" all review details collected ")

                    # Trying to save the data
                    try:
                        reviews_full_df.to_csv(searchstring+'review_data.csv')
                        logging.info(' review Data Downloaded in CSV format')
                    except Exception as e:
                        logging.exception(e)

                    # trying to store the data in mongodb
                    try:
                        logging.info('trying to store data in Mongodb')
                        client = pymongo.MongoClient("mongodb+srv://sahil5723:NEWlife123@cluster0.1bbad.mongodb.net/?retryWrites=true&w=majority")
                        df1 = reviews_full_df.to_json(orient='records')
                        df2 = json.loads(df1)
                        database = client['project']  # database name
                        collection = database['gp_reviews_test']  # creating document
                        collection.insert_many(df2)
                    except Exception as e:
                        logging.info(" Database issue ")
                        logging.exception(e)

                    content = []
                    for i in reviews_full:
                        for j in i:
                            content.append(j['content'])  # storing the reviews only

                    common_content = ''.join(str(content))
                    logging.info('review content extracted ')


                    # Removing unwanted symbols incase if exists
                    content_string = re.sub("[^A-Za-z" "]+", " ", common_content).lower()
                    spell = Speller(lang='en')
                    spelling_correction = []
                    for i in content_string.split():  # spell checking
                        spelling_correction.append(spell(i))

                    logging.info('Spell check done')
                    lemmatizer = WordNetLemmatizer()
                    lemmas = []
                    for i in spelling_correction:
                        lemmas.append(lemmatizer.lemmatize(i))

                    filtered_sentence = []
                    stop_words = set(stopwords.words('english'))  # stop words removal
                    for w in lemmas:
                        if w not in stop_words:
                            filtered_sentence.append(w)
                    logging.info('Stop words removed')

                    bigrams_list = list(nltk.bigrams(filtered_sentence))
                    dictionary2 = [' '.join(tup) for tup in bigrams_list]
                    from sklearn.feature_extraction.text import CountVectorizer
                    vectorizer = CountVectorizer(ngram_range=(2,2))
                    bag_of_words = vectorizer.fit_transform(dictionary2)
                    # vectorizer.vocabulary_

                    sum_words = bag_of_words.sum(axis=0)
                    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
                    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

                    words_dict = dict(words_freq)
                    WC_height = 1000
                    WC_width = 1500
                    WC_max_words = 100
                    wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width)

                    wordCloud.generate_from_frequencies(words_dict)
                    plt.figure(4)
                    plt.title('Most frequently occurring bigrams connected by same colour and font size')
                    plt.imshow(wordCloud, interpolation='bilinear')
                    plt.axis("off")
                    image_path = os.path.join(appp.config['IMG_FOLDER'], searchstring+'_review' + '.png')
                    plt.savefig(image_path)
                    #img_file = os.listdir(appp.config['IMG_FOLDER'])[0]
                    #full_filename = os.path.join(appp.config['IMG_FOLDER'], img_file)
                    full_filename = image_path
                    logging.info(' review word cloud created successfully')
                except Exception as e:
                    logging.info('Error occured in review analysis')
                    logging.exception(e)

            return render_template('wordcloudoutput.html', user_image= full_filename)
        except Exception as e:
            logging.exception(e)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    # appp.run(host='0.0.0.0', port=8000)
    appp.run()
