from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/response', methods=['POST'])
def response():
    fname = request.form.get("fname")
    note = request.form.get("note")
    return render_template("index.html", name=fname, note=note)

if __name__ == '__main__':
    app.run(debug=True, port=5000)