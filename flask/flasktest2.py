from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)

@app.route('/')
def gen_info():
    return render_template("login.html")

@app.route('/train/<name>')
def train(name):
   return f"the project id and api key is: \n {name}"

@app.route('/img', methods = ['POST', 'GET'])
def show_img():
   import base64
   if request.method == 'POST':
      filename = request.form['filename']
      e = filename.split(".")[-1]

      img = open(filename, 'rb').read()

      data = base64.b64encode(img).decode()

      src = "data:image/{e};base64,{data}".format(e=e, data=data)

      return render_template('example.html', url=src)

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      project_id = request.form['pid']
      api_key = request.form['api']
      return redirect(url_for('train', name = project_id + '.' + api_key))
   else:
      project_id = request.form['pid']
      api_key = request.form['api']
      return redirect(url_for('train', name = project_id + '.' + api_key))

if __name__ == '__main__':
   app.run(debug = True)