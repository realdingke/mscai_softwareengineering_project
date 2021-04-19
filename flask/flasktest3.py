from flask import Flask, redirect, url_for, request, render_template, jsonify

app = Flask(__name__)


@app.route('/')
def gen_info():
    return render_template("main.html")


@app.route('/train/<name>')
def train(name):
    return f"the project id and api key is: \n {name}"


@app.route('/login', methods=['POST', 'PUT'])
def login():
    if request.method == 'POST':
        project_id = request.form['pid']
        api_key = request.form['api']
        # seq_info = {'project_ID':project_id, 'API_key':api_key}
        # return render_template("print_dict.html", result=jsonify(seq_info))
        return "The results are          " + \
            f"project_ID: {project_id}           " + \
            f"API_key: {api_key}         " + \
            "<b><a href = '/'>click here to return to main page</a></b>"
    else:
        project_id = request.form['pid']
        api_key = request.form['api']
        return redirect(url_for('train', name=project_id + '.' + api_key))


if __name__ == '__main__':
    app.run(debug=True)
