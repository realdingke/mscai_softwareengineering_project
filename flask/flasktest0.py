from flask import Flask, request, jsonify, send_from_directory

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
    file_name = 'D:\Engineering Files\Computing_Msc_AI\software_engineering\example\summary_hrnet_18_50_allframes_test.xlsx'
    return send_from_directory('folder-path', file_name, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)