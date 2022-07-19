#import numpy as np
from flask import Flask, request, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    y = []
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    y.append(review)
        
    return "".join(y)

app = Flask(__name__)
cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    message=request.form['message']
    
    transformed_text=transform_text(message)
    
    vector_input=cv.transform([transformed_text])
    
    result=model.predict(vector_input)[0]

    if result==1:    
        return render_template('index.html', prediction_text='Message is likely a spam message')
    else:
        return render_template('index.html', prediction_text='Message is not a spam message')

if __name__ == "__main__":
    app.run(debug=True)