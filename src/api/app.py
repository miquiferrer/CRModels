import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from api.serve_model import get_model_api  # see part 1.

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()

app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default
model_api = get_model_api()


# default route
@app.route('/')
def index():
    return "Congratulations! Flask is properly running!"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


# API route
@app.route('/api', methods=['POST'])
def api():
    input_data = request.json
    output_data = model_api(input_data)
    response = jsonify(output_data)
    return response


# Run the web service
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
