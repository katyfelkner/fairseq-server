#!/usr/bin/env python
"""
Serves a Fairseq translation model using Flask HTTP server
Ported to Fairseq from RTG.serve by Thamme Gowda.
"""
import logging
import os
import sys
import platform
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import fairseq
import torch

import flask
from flask import Flask, request, send_from_directory, Blueprint

from fairseq.models.transformer import TransformerModel

torch.set_grad_enabled(False)
FLOAT_POINTS = 4
exp = None
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

bp = Blueprint('nmt', __name__, template_folder='templates')

sys_info = {
    'Fairseq Version': fairseq.__version__,
    'PyTorch Version': torch.__version__,
    'Python Version': sys.version,
    'Platform': platform.platform(),
    'Platform Version': platform.version(),
    'Processor':  platform.processor(),
    'GPU': '[unavailable]',
}



def render_template(*args, **kwargs):
    return flask.render_template(*args, environ=os.environ, **kwargs)


def jsonify(obj):

    def _jsonify(ob):
        if ob is None or isinstance(ob, (int, bool, str)):
            return ob
        elif isinstance(ob, float):
            return round(ob, FLOAT_POINTS)
        elif isinstance(ob, dict):
            return {key: _jsonify(val) for key, val in ob.items()}
        elif isinstance(ob, list):
            return [_jsonify(it) for it in ob]
        elif isinstance(ob, np.ndarray):
            return _jsonify(ob.tolist())
        else:
            logging.warning(f"Type {type(ob)} maybe not be json serializable")
            return ob

    obj = _jsonify(obj)
    return flask.jsonify(obj)

@bp.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(bp.root_path, 'static', 'favicon'), 'favicon.ico')


def attach_translate_route(cli_args):
    global model
    model = TransformerModel.from_pretrained(cli_args.pop("chkpt_path"), checkpoint_file='checkpoint_best.pt', data_name_or_path='data-bin/')

    @bp.route("/translate", methods=["POST", "GET"])
    def translate():
        if request.method not in ("POST", "GET"):
            return "GET and POST are supported", 400
        if request.method == 'GET':
            sources = request.args.getlist("source", None)
        else:
            sources = (request.json or {}).get('source', None) or request.form.getlist("source")
            if isinstance(sources, str):
                sources = [sources]
        if not sources:
            return "Please submit 'source' parameter", 400

        # Check for ASCII only
        if not all(s.isascii() for s in sources):
            return "Only ASCII characters are accepted", 400

        prep = request.args.get('prep', "True").lower() in ("true", "yes", "y", "t")

        remap = {} # original: standardized
        if prep:
            # whitespace tokenize
            tokenized = [sent.strip().split() for sent in sources]
            var_counter = 0
            for source in tokenized:
                for s in source:
                    if s not in ['(', ')', '*', '+', '-', '/', '-1', '[', ']', '{', '}'] and s not in remap.keys():
                        std_name = "a_" + str(var_counter)
                        var_counter += 1
                        remap[s] = std_name

            remapped = [' '. join([remap[s] if s in remap.keys() else s for s in sent]) for sent in tokenized]
            print(remapped)

        translations = []
        unremap = {v: k for k, v in remap.items()}
        for source in remapped:
            translated = model.translate(source)
            print(translated)
            if prep:
                # replace remapped variables with original variables
                unremapped = ' '. join([unremap[s] if s in unremap.keys() else s for s in translated.strip().split()])
            translations.append(unremapped)



        res = dict(source=sources, translation=translations)
        return jsonify(res)


    """ @bp.route("/conf.yml", methods=["GET"])
    def get_conf():
        conf_str = exp._config_file.read_text(encoding='utf-8', errors='ignore')
        return render_template('conf.yml.html', conf_str=conf_str)

    @bp.route("/about", methods=["GET"])
    def about():
        def_desc = "Model description is unavailable; please update conf.yml"
        return render_template('about.html', model_desc=exp.config.get("description", def_desc),
                               sys_info=sys_info) """

# TODO need to update to useful fairseq args and/or sensible defaults
def parse_args():
    parser = ArgumentParser(
        prog="fairseq-server",
        description="Deploy an RTG model to a RESTful server",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("chkpt_path", help="Path to model checkpoints", type=str)
    parser.add_argument("-d", "--debug", action="store_true", help="Run Flask server in debug mode")
    parser.add_argument("-p", "--port", type=int, help="port to run server on", default=6060)
    parser.add_argument("-ho", "--host", help="Host address to bind.", default='0.0.0.0')
    parser.add_argument("-b", "--base", help="Base prefix path for all the URLs")
    parser.add_argument("-msl", "--max-src-len", type=int, default=250,
                        help="max source len; longer seqs will be truncated")
    args = vars(parser.parse_args())
    return args


cli_args = parse_args()
attach_translate_route(cli_args)
app.register_blueprint(bp, url_prefix=cli_args.get('base'))
if cli_args.pop('debug'):
    app.debug = True

# register a home page if needed
if cli_args.get('base'):
    @app.route('/')
    def home():
        return render_template('home.html', demo_url=cli_args.get('base'))


def main():
    app.run(port=cli_args["port"], host=cli_args["host"])
    # A very useful tutorial is found at:
    # https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3


if __name__ == "__main__":
    main()
