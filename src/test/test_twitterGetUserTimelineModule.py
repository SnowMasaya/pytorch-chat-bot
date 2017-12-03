# -*- coding: utf-8 -*-
from __future__ import absolute_import

from __future__ import unicode_literals
from unittest import TestCase
from data.twitter_get_usr_timeline_module import TwitterGetUserTimelineModule
import os


class TestTwitterGetUserTimelineModule(TestCase):
    def test_twitter_method(self):
        APP_PATH = os.path.dirname(__file__)
        self.twitter_get_usr_timeline_module = TwitterGetUserTimelineModule(
            config_file=APP_PATH + '/../data/config/environment_twitter.yml')
        for k, v in self.twitter_get_usr_timeline_module.twitter_txt_dict.items():
            params = {"screen_name": k, "exclude_replies": False,
                      "count": 1}
            req = self.twitter_get_usr_timeline_module.twitter.get(
                self.twitter_get_usr_timeline_module.url, params=params)
            self.twitter_get_usr_timeline_module.twitter_method(req, dict_flag=True,
                                                     dict_value=v)
        # SQLite
        self.twitter_get_usr_timeline_module.conn.commit()
        self.twitter_get_usr_timeline_module.conn.close()


