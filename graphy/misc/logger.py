import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
    def write(self, message):
        self.terminal.write(message)
        with open(self.filename, "a") as log:
            log.write(message)
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
