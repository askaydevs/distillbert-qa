import flask
from flask import request
from model import Model

model = Model('model')

app = flask.Flask(__name__)

@app.route('/')
def index():

    if request.args:

        context = request.args["context"]
        question = request.args["question"]

        answer = model.predict(context, question)
        print(answer["answer"])

        return flask.render_template('index.html', question=question, answer=answer["answer"])
    else:
        return flask.render_template('index.html')
