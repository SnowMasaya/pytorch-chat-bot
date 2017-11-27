#!/usr/bin/env python
#-*- coding:utf-8 -*-
#!/usr/bin/python3

import sqlite3
import MeCab
import yaml
from collections import namedtuple
import os
import neologdn
APP_PATH = os.path.dirname(__file__)


class SqliteTwitter(object):
    """
    Twitter Save to the SQLite
    """
    def __init__(self, config_file: str=APP_PATH + "/config/mecab.yml",
                 sqlite_file: str=APP_PATH + '/../../data/processed/twitter_data.db'):
        """
        Initial Setting
        Get the mecab dict by the yaml
        """
        Mecab = namedtuple("Mecab", ["dict"])
        config_file = config_file

        with open(config_file, encoding="utf-8") as cf:
            e = yaml.load(cf)
            mecab = Mecab(e["mecab"]["dict"])

        self.tagger = MeCab.Tagger("-Owakati -d %s" % mecab.dict)
        conn = sqlite3.connect(sqlite_file)
        self.cur = conn.cursor()

    def call_sql(self):
        """
        call SQlite and save the twitter in the SQLite
        """
        self.cur.execute("""SELECT source_txt, replay_txt FROM ms_rinna;""")
        for source_txt, replay_txt in self.cur.fetchall():
            with open('source_replay_twitter_data.txt', 'a') as f:

                if replay_txt.find('https://t.co') >= 0:
                    replay_txt = neologdn.normalize(replay_txt).replace("\n", "")
                else:
                    replay_txt = self.tagger.parse(neologdn.normalize(replay_txt)).replace("\n", "")
                if source_txt.find('https://t.co') >= 0:
                    target_txt = neologdn.normalize(source_txt).replace("\n", "")
                else:
                    target_txt = self.tagger.parse(neologdn.normalize(source_txt)).replace("\n", "")

                f.write(replay_txt + '\t' +
                        target_txt + '\n'
                        )


if __name__ == '__main__':
    sqlite_twitter = SqliteTwitter()
    sqlite_twitter.call_sql()