# -*- coding: utf-8 -*-
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import os
from data.multi_thread_producer_consumer_twitter_async import ProducerConsumerThreadTwitterAsync
from data.multi_thread_producer_consumer_twitter_async import SEED_DOMAIN_LIST_SIZE
from os import path
import argparse
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
APP_ROOT = path.dirname(path.abspath( __file__ ))

"""
This script for parallel command multi thread program
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Twitter Producer consumer')
    parser.add_argument("-q", "--queue_size", metavar="queue_size",
                        type=int, default=30,
                        dest="queue_size", help="set the queue size ")
    args = parser.parse_args()
    SEED_DOMAIN_LIST_SIZE = args.queue_size
    producerConsumerThreadTwitterAsync = ProducerConsumerThreadTwitterAsync()
    loop = asyncio.get_event_loop()

    #Producer
    loop.create_task(producerConsumerThreadTwitterAsync.producer_run())

    #Consumer
    loop.create_task(producerConsumerThreadTwitterAsync.consumer_run())
    loop.run_forever()
    loop.close()

