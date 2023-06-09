Case Study 2
==============================
Introduction
Football, as we all know, is an international sport that is particularly popular in Europe. The market value of footballers is one of the most intriguing aspects of football when it comes to federations, clubs, and players. The projection of market valuations of players has always been one of the most important considerations for a football club, and it can be influenced by a variety of factors. Our analysis includes a variety of athlete characteristics such as age, height, left or right foot, number of goals per season, and international impact.

This is a web application designed to show the project structure for a machine learning model deployed using flask. This project features a machine learning model that has been trained to predict the value of players.



## Installation

First clone the repo locally.
~~~bash
git clone https://github.com/Pranshu1993/Transfer_Market_Values_Prediction.git
~~~

Create a new virtual environment in the project directory.
~~~bash
python3 -m venv ./venv
~~~

Activate the virtual environment.
~~~bash
source venv/bin/activate
~~~

While in the virtual environment, install required dependencies from `requirements.txt`.

~~~bash
pip install -r ./requirements.txt
~~~

Now we can deploy the web application via
~~~bash
python app.py
~~~

and navigate to `http://127.0.0.1:5000/` to see it live. On this page, a user can then submit text into the text 
field and receive predictions from the trained model and determine how much the player is worth with your own stats.


The application may then be terminated with the following commands.
~~~bash
$ ^C           # exit flask application (ctrl-c)
$ deactivate   # exit virtual environment
~~~
