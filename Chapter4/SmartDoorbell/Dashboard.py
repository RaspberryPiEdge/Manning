import cherrypy
from jinja2 import Environment, FileSystemLoader
import json
import os
from LogRecord import LogRecord


class Dashboard:
    def __init__(self, log_path):
        self.env = Environment(loader=FileSystemLoader('templates'))
        self.log_record = LogRecord(log_path)

    @cherrypy.expose
    def index(self):
        tmpl = self.env.get_template('index.html')
        data = self.log_record.get_record()
        return tmpl.render(data=data)

    @cherrypy.expose
    def getdata(self, datetime=None):
        data = self.log_record.get_record()
        for item in data:
            if item['datetime'] == datetime:
                return json.dumps({'name': item['name']})

        return json.dumps({'name': 'Data not found'})


if __name__ == '__main__':
    
    log_path = 'logs/recognition_timestamps.csv'
    
    conf = {
        '/images': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': os.path.abspath('./images')
        }
    }
    cherrypy.config.update({'server.socket_host': '10.0.0.62', 'server.socket_port': 8080})
    cherrypy.quickstart(Dashboard(log_path), '/', conf)
