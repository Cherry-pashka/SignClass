from flask import Flask
app = Flask(__name__)

@app.route("/&lt;username&gt;", methods=['GET'])
def index(username):
    return "Hello, %s!" % username

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4567)