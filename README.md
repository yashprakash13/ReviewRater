## Review Rater 
A simple web app prototype that uses sentiment analysis to rate text reviews on a scale of 1 to 5.

### Setup on your machine
* This repository uses Git Large File Storage(git-lfs) to host large binary files(.pickle, .h5, etc). If you don't have git lfs installed on your system, go ahead install it from here: https://git-lfs.github.com/

* Next, inside the empty folder in which you want to clone, run the command 
`git lfs install`
* That's it, now clone this repo with the usual command:
`git clone https://github.com/yashprakash13/ReviewRater.git`
* To run the web app, after the lfs objects have been prooperly downloaded into your local folder, type `python main.py`.
* Go to http://127.0.0.1:5000/ and test the app with the following credentials:
  * email: hello@world.com
  * password: 1234567
  * Payment details : enter anything in the boxes and use the demo card no 4242 4242 4242 4242
  * type a review and click Predict!
  
### Tools used
This project uses:
* Flask for the web app
* Firebase for authentication of users (registration and login)
* Stripe for payment demo and
* Sentiment analysis machine learning models for rating the review (Folders Model, Model2 and Model3) 

### The project in action
![](https://github.com/yashprakash13/ReviewRater/blob/master/screenshots/1.png) ![](https://github.com/yashprakash13/ReviewRater/blob/master/screenshots/2.png)
![](https://github.com/yashprakash13/ReviewRater/blob/master/screenshots/3.png) ![](https://github.com/yashprakash13/ReviewRater/blob/master/screenshots/4.png) ![](https://github.com/yashprakash13/ReviewRater/blob/master/screenshots/5.png)
