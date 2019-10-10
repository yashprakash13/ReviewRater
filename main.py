import pyrebase
from flask import *
import os
import stripe

app = Flask(__name__)

config = {
    "apiKey": "AIzaSyCDHxr_7dUT5oeA6W1sO_UZ4-mSw1T97Ew",
    "authDomain": "inosis-46c45.firebaseapp.com",
    "databaseURL": "https://inosis-46c45.firebaseio.com",
    "projectId": "inosis-46c45",
    "storageBucket": "",
    "messagingSenderId": "992130889346",
    "appId": "1:992130889346:web:c6109b00ed20059b1d269b",
    "measurementId": "G-FZB6CCYNHQ"
}

firebase = pyrebase.initialize_app(config)

auth = firebase.auth()

db = firebase.database()

user_email = ""


@app.route('/', methods=['GET', 'POST'])
def loginUser():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            auth.sign_in_with_email_and_password(email, password)
            user_email = email
            return render_template('home.html')  # Login Success!
        except:
            # Login Failed!
            return render_template('index.html', error="error")

    return render_template('index.html')


@app.route('/gotoreg/', methods=['GET', 'POST'])
def redirectToReg():
    return render_template('register.html')


@app.route('/gotologin/', methods=['GET', 'POST'])
def redirectToLogin():
    return render_template('index.html')


@app.route('/reg', methods=['GET', 'POST'])
def signUpUser():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            auth.create_user_with_email_and_password(email, password)
            user_email = email

            return render_template('home.html')
        except:
            return render_template('register.html', error="error")

    return render_template('register.html')


@app.route('/homecheckout', methods=['GET', 'POST'])
def showModel():

    # amount in cents
    # amount = 2900

    # customer = stripe.Customer.create(
    #     email='sample@customer.com',
    #     source=request.form['stripeToken']
    # )

    # stripe.Charge.create(
    #     customer=customer.id,
    #     amount=amount,
    #     currency='usd',
    #     description='Flask App Charge'
    # )

    return render_template('model.html')


@app.route('/predict', methods=['POST'])
def predict():
    review = str(request.form['message'])
    results = []

    # 1. NA3- Model 1
    import sys
    from gensim.models import Word2Vec
    import os
    x = 'Model/300features_40minwords_10context_newWithSentimentColumn'
    model = Word2Vec.load(x)

    import pickle
    y = 'Model/RF_W2V_Classifier.pickle'
    f = open(y, 'rb')
    forest = pickle.load(f)
    f.close()

    sys.path.append('Model')
    import clean_text_vector as ctv

    review_vec = ctv.getVec(review, model, 300)
    res = forest.predict(review_vec)
    results.append(res[0].astype(float))

    # print("Predicting....just a sec...")

    # 2. CVModel- Model 2
    from sklearn.externals import joblib
    a = 'Model2/CountVectorizerModel.pkl'
    model = joblib.load(a)

    pr = model.predict([review])[0]
    results.append(pr.astype(float))

    # print("Done second model...")

    # 3. Finally- Model 3
    from keras.models import load_model
    import numpy as np
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(review)
    seq = tokenizer.texts_to_sequences(review)
    seq_pad = pad_sequences(seq, maxlen=500)

    b = 'Model3/kerasModel.h5'
    model = load_model(b)
    pr = model.predict(seq_pad)
    pr = np.argmax(pr, axis=1)
    u, indices = np.unique(pr, return_inverse=True)
    axis = 0
    results.append(u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(
        pr.shape), None, np.max(indices) + 1), axis=axis)])

    sentiment = int(max(results, key=results.count))

    conf = (results.count(sentiment))/3  # confidence level of prediction

    toShow = 'Rating should be: '
    # print('Rating should be: ', end='')

    stance = ""

    if sentiment == 1:
        if conf > 0.66:
            toShow = toShow + '5 stars'
        else:
            toShow = toShow + "4 stars"
        stance = "Positive"
    elif sentiment == 0:
        if conf > 0.66:
            toShow = toShow + '3 stars'
        else:
            toShow = toShow + "2.5 stars"
        stance = "Neutral"
    else:
        if conf > 0.66:
            toShow = toShow + '1 star'
        else:
            toShow = toShow + "2 stars"
        stance = "Negative"

    # print("Your stance is: ", stance)
    toShow = toShow + " and your stance is " + stance
    return render_template('model.html', pred=toShow)


if __name__ == '__main__':
    app.run()

# email = input("Email")
# password = input("Password")
# #user = auth.create_user_with_email_and_password(email, password)
# user = auth.sign_in_with_email_and_password(email, password)

# auth.get_account_info(user['idToken'])

# # auth.send_email_verification(user['idToken'])
