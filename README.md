# News Topic Classification
This is a Python project for classifying news articles into different topics using machine learning and deep learning techniques. The project includes a web scraper that collects news articles from several popular news websites, a machine learning model based on SVM, and a deep learning model based on a convolutional neural network (CNN).

Dependencies
To run this project, you'll need the following dependencies installed:

<ul>
<li>Python 3.10.9</li>
<li>Beautiful Soup 4</li>
<li>Requests</li>
<li>NumPy</li>
<li>Pandas</li>
<li>Scikit-learn</li>
<li>TensorFlow</li>
<li>Keras</li>

</ul>
You can install these dependencies by running the following command:


<code>
```pip install -r requirements.txt
```</code>

The scraper.py file contains the web scraper code that collects news articles from several popular news websites, including CNN, BBC, and Reuters. The scraper collects articles from several different topics, including politics, business, and technology.

To run the scraper, simply run the following command:

```python scraper.py```

The scraper will collect the articles and save them to a CSV file called articles.csv.

Machine Learning Model
The ML_Model.py file contains the code for training and testing the machine learning model based on SVM. The model uses a bag-of-words approach to represent the articles as feature vectors and trains an SVM classifier on the feature vectors.

To train the model, run the following command:

Copy code
python ML_Model.py train
To test the model, run the following command:

bash
Copy code
python ML_Model.py test
The model will be trained and tested on the articles in articles.csv, and the results will be saved to a CSV file called ml_results.csv.

The trained SVM model and vectorizer are saved in svm_model.pkl and svm_vectorizer.pkl, respectively.

Deep Learning Model
The DL_Model.py file contains the code for training and testing the deep learning model based on a convolutional neural network (CNN). The model uses word embeddings to represent the articles as sequences of vectors and trains a CNN on the sequences.

To train the model, run the following command:

Copy code
python DL_Model.py train
To test the model, run the following command:

bash
Copy code
python DL_Model.py test
The model will be trained and tested on the articles in articles.csv, and the results will be saved to a CSV file called dl_results.csv.

The trained deep learning model and tokenizer are saved in dl_model.pkl and dl_tokenizer.pkl, respectively. The maximum sequence length used by the tokenizer is saved in dl_max_length.pkl.

Streamlit App
The app.py file contains a Streamlit app that allows you to classify news articles using either the SVM or deep learning model. The app loads the trained models and vectorizers and allows you to enter your own news article to classify.

To run the app, simply run the following command:

arduino
Copy code
streamlit run app.py
The app will start on port 8501, and you can access it by navigating to http://localhost:8501 in your web browser.

Docker
You can also run this project in a Docker container. To build the Docker image, run the following command:

sql
Copy code
docker build -t news-classifier .
To run the Docker container, run the following command:

arduino
Copy code
docker run -p 8501:8501 news-classifier
Make sure that you've saved all of the required files (including app.py) in the same directory as your Dockerfile.

⋅⋅*
