from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, request, render_template
import os
import pickle
import re
import json
import entry_point, paths, train, mctrack
from paths import CLIENT_DATA_PATH
from clean import clean_files_a
from lib.opts import opts

app = Flask(__name__)
opt = opts().init()


# @app.route("/")
# def test0_api():
#    return jsonify(data0="Yes, this is DK's first try")

@app.route('/')
def gen_info():
    return render_template("main.html")


@app.route('/train_html')
def train_html():
    file_name_path = paths.PATHS_OBJ_PATH
    with open(file_name_path, 'rb') as f:
        paths_loader = pickle.load(f)
    seqs_name_path = paths_loader.SEQS_NAME_PATH
    with open(seqs_name_path, 'rb') as f:
        seqs_name_dict = pickle.load(f)
    seqs = seqs_name_dict['labeled_seqs']
    return render_template("train.html", seqs=seqs)


@app.route('/track_html')
def track_html():
    models_name = [direc for direc in os.listdir(paths.MODEL_DIR_PATH)]
    file_name_path = paths.PATHS_OBJ_PATH
    with open(file_name_path, 'rb') as f:
        paths_loader = pickle.load(f)
    seqs_name_path = paths_loader.SEQS_NAME_PATH
    with open(seqs_name_path, 'rb') as f:
        seqs_name_dict = pickle.load(f)
    seqs = seqs_name_dict['labeled_seqs'] + seqs_name_dict['empty_seqs']
    return render_template("track.html", models=models_name, seqs=seqs)


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


@app.route('/login', methods=['POST', 'PUT'])
def login():
    opt.gen_info = True
    if request.form['pid'] != '':
        opt.project = request.form['pid']
    if request.form['api'] != '':
        opt.api = request.form['api']
    # opt.email = request.form['email']
    results_dict = entry_point.main(opt)
    opt.gen_info = False
    return render_template("gen_info.html", results=results_dict)


@app.route('/train', methods=['POST', 'PUT'])
def train_track():
    # Entry_point
    opt.train_track = True
    train_seq = request.form.getlist('train_seq')
    if len(train_seq) > 0:
        opt.dataset_selection = train_seq
    if request.values.get('rand_split') == 'True':
        opt.rand_split = True
    else:
        opt.rand_split = False
    if request.form['rseed'] != '':
        opt.rseed = int(request.form['rseed'])
    if request.form['split_perc'] != '':
        split_perc = request.form['split_perc']
        opt.split_perc = [[float(sp.strip()) for sp in split_perc.split()]]
    if request.form['name'] != '':
        opt.json_name = request.form['name']
    results_dict_train_track = entry_point.main(opt)
    # Train
    if request.form['lr'] != '':
        opt.lr = float(request.form['lr'])
    if request.form['batch'] != '':
        opt.batch_size = int(request.form['batch'])
    if request.form['epoch'] != '':
        opt.num_epochs = int(request.form['epoch'])
    if request.form['drop'] != '':
        opt.lr_step = request.form['drop']
    if request.form['exp_id'] != '':
        opt.exp_id = request.form['exp_id']
        opt.exp_name = opt.exp_id
    model_type = request.values.get('model_type')
    if model_type != '-- Choose --':
        opt.arch = model_type
    if request.values.get('add_test') == 'True':
        opt.add_test_dataset = True
    else:
        opt.add_test_dataset = False
    if request.values.get('plot_loss') == 'True':
        opt.plot_loss = True
    else:
        opt.plot_loss = False
    if request.values.get('save_time') == 'True':
        opt.save_time = True
    else:
        opt.save_time = False
    # if request.form[] != '':
    #     opt.num_iters = request.form[]

    # coped from train main function
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '0, 1'
    # opt = opts().parse()
    # automatically identify reid_cls_ids
    file_name_path = paths.PATHS_OBJ_PATH
    opt.load_model = ''
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'rb') as f:
            paths_loader = pickle.load(f)
        # automatically identify reid_cls_ids
        id2cls_path = os.path.join(paths_loader.TRAIN_DATA_PATH, 'id2cls.json')
        if os.path.isfile(id2cls_path):
            with open(id2cls_path, 'r') as f:
                data = json.load(f)
            cls_ids_ls = list(data.keys())
            id_str = ", ".join(cls_ids_ls)
            opt.reid_cls_ids = id_str
    results_dict_train = train.train(opt)

    # MCTrack
    opt.load_model = '/home/user/exp/mot/' + opt.exp_id + f"/model_{opt.num_epochs}.pth"
    output_format = request.values.get('output_format')
    if output_format != '-- Choose --':
        opt.output_format = output_format
    # if request.form['exp_name'] != '':
    #     opt.exp_name = request.form['exp_name']
    if request.values.get('track_time') == 'True':
        opt.save_track_time = True
    else:
        opt.save_track_time = False
    # if request.form[] != '':
    #     opt.conf_thres = request.form[]

    # Coped from mctrack main function
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 0
    # opt = opts().init()
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'rb') as f:
            path_object = pickle.load(f)
        if not opt.val_mot16:
            data_root = path_object.TEST_DIR_NAME_PATH
            seqs_str = os.listdir(data_root)
            seqs_str = '  \n'.join(seqs_str)

    # convert whitespace in between filename into '_'
    pattern = '(?<=\w)\s(?=\w)'
    try:
        seqs_str = re.sub(pattern, '_', seqs_str)
    except:
        seqs_str = seqs_str

    seqs = [seq.strip() for seq in seqs_str.split()]
    # seqs = [string.replace('_', ' ') for string in seqs]

    results_dict_track = mctrack.track(opt,
                                       data_root=data_root,
                                       seqs=seqs,
                                       exp_name=opt.exp_name,
                                       show_image=False,
                                       save_images=False,
                                       save_videos=False)
    opt.train_track = False
    return render_template("train_result.html",
                           opt=opt,
                           results_track=results_dict_track,
                           results_train=results_dict_train,
                           results_train_track=results_dict_train_track,
                           )


@app.route('/track', methods=['POST', 'PUT'])
def track():
    opt.track = True
    videos = request.form.getlist('videos')
    if len(videos) > 0:
        opt.tracking_video_selection = [videos]
    output_format = request.values.get('output_format')
    if output_format != '-- Choose --':
        opt.output_format = output_format
    model_type = request.values.get('model_type')
    if model_type != '-- Choose --':
        opt.specified_model = model_type
    if request.values.get('visual') == 'True':
        opt.visual = True
    else:
        opt.visual = False
    _ = entry_point.main(opt)
    opt.track = False
    return render_template("track_result.html", opt=opt)


@app.route('/clean', methods=['GET'])
def clean():
    clean_files_a()
    return "<b><a href = '/'>click here to return to main page</a></b>"


@app.route('/restore', methods=['POST', 'PUT'])
def restore():
    opt.restore = True
    _ = entry_point.main(opt)
    opt.restore = False
    return "<b><a href = '/'>click here to return to main page</a></b>"


@app.route('/clean_all', methods=['GET'])
def clean_all():
    opt.clean_model = True
    _ = entry_point.main(opt)
    opt.clean_model = False
    return "<b><a href = '/'>click here to return to main page</a></b>"
# @app.route('/display_pic', methods=['POST', 'PUT'])
# def display_pics():
#     pics =
#     return pics # String to display pics


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
