from webapi.app import app as server

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=8081,debug=True)
