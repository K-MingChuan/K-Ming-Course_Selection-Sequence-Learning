import ctypes
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import sys


def parse_path(path):
    result = urlparse(path)
    return {
        'path': result[2],
        'query': dict((key, vals[0]) for key, vals in parse_qs(result[4]).items())
    }


class KMingServer(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server,
                 recommender):
        self.recommender = recommender
        super().__init__(request, client_address, server)

    def _send_headers(self, state_code):
        self.send_response(state_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def _respond(self, state_code, msg, data):
        self._send_headers(state_code)
        self.wfile.write(bytes(json.dumps({
            'message': msg,
            'data': data
        }), 'UTF-8'))

    def do_GET(self):
        params = parse_path(self.path)
        try:
            if params['path'] == '/recommendations':
                if 'id' in params['query']:
                    student_id = params['query']['id']
                    courses = self.recommender(student_id)
                    self._respond(200, 'success', {'student_id': student_id, 'courses': courses})
                else:
                    self._respond(400, 'the param "id" should be given.', None)
        except Exception as err:
            self._respond(400, 'Error occurs: ' + str(err), None)


def run(server_class=HTTPServer, handler_class=None, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


if __name__ == "__main__":
    from sys import argv

    # if the script is not run in the administrator privilege, pop up the UAC dialog to seek to elevate
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

    stud_recommender = lambda student_id: ['物件導向', '豐緒的課']
    port = int(argv[1]) if len(argv) == 2 else 5000

    run(port=port, handler_class=lambda request, client_address, server: KMingServer(request, client_address, server,
                                                                                     stud_recommender))

