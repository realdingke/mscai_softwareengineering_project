import sys
sys.path.append("..")

from flask import Flask, request, jsonify, send_from_directory
from src.paths import CLIENT_DATA_PATH
# from src import entry_point
# from src.lib.opts import opts
# from src.lib import logger

app = Flask(__name__)

#@app.route("/")
#def test0_api():
#    return jsonify(data0="Yes, this is DK's first try")

def exception_handler(func):
  def wrapper(*args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_code = getattr(e, "code", 500)
        logger.exception("Service exception: %s", e)
        r = dict_to_json({"message": e.message, "matches": e.message, "error_code": error_code})
        return Response(r, status=error_code, mimetype='application/json')
  # Renaming the function name:
  wrapper.__name__ = func.__name__
  return wrapper

@app.route("/path1")
@exception_handler
def func1():
    return jsonify(data0="Yes, this is DK's first try")

@app.route("/path2")
@exception_handler
def func2():
    return jsonify(data1="Yes, this is DK's second try")

@app.route("/path3")
@exception_handler
def return_a_table():
    file_path = 'E:/GroupProject/mscai_softwareengineering_project/flask/'
    file_name = 'summary_hrnet_18_50_allframes_test.xlsx'
    return send_from_directory(file_path, file_name, as_attachment=True)

@app.route('/path4', methods=['GET', 'POST'])
@exception_handler
def get_and_return():
    a = dict()
    if request.form['gen_info'] == 'true':
        # a.update(dict(jsonify(test='test')))
        a.update({'test': 'test'})
    if request.form['print_path'] == 'true':
        # a.update(dict(jsonify(info=CLIENT_DATA_PATH)))
        a.update({'info': CLIENT_DATA_PATH})
    if len(a) > 0:
        return jsonify(a)

@app.route('/path5', methods=['GET', 'POST'])
@exception_handler
def gen_info():
    # opt = opts().init()
    if request.form['project_id'] != '':
        opt.project = request.form['project_id']
    if request.form['api_key'] != '':
        opt.api = request.form['api_key']
    if request.form['gen_info'] == 'true':
        opt.gen_info = True
        results_dict = entry_point.main(opt)
        return jsonify(results_dict)


if __name__ == "__main__":
    app.run(debug=True)