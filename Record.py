import datetime
from os import mkdir


class Record:
    def __init__(self):
        self.log = ""
        self.stamp = datetime.now().strftime("%m_%d_%H_%M")
    def print(self, string):
        print(string)
        self.log+=('\n')
        self.log+=(string)

    def dump(self):
        mkdir("./simResults/{}".format(self.stamp))
        log_txt = open("./simResults/{}/log.txt".format(self.stamp),"w+")
        log_txt.write(self.log)
