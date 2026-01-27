"""
Deepfake Audio - Logger
-----------------------
A simple logging utility for tracking training progress.
Supports writing to a text file and optionally sending status updates to Slack.

Authors:
    - Amey Thakur (https://github.com/Amey-Thakur)
    - Mega Satish (https://github.com/msatmod)

Repository:
    - https://github.com/Amey-Thakur/DEEPFAKE-AUDIO

Release Date:
    - February 06, 2021

License:
    - MIT License
"""

import atexit
import json
from datetime import datetime
from threading import Thread
from urllib.request import Request, urlopen
from typing import Optional

_format = "%Y-%m-%d %H:%M:%S.%f"
_file = None
_run_name = None
_slack_url = None


def init(filename: str, run_name: str, slack_url: Optional[str] = None):
    """Initializes the logger, opening the log file and setting up Slack integration."""
    global _file, _run_name, _slack_url
    _close_logfile()
    _file = open(filename, "a")
    _file.write("\n-----------------------------------------------------------------\n")
    _file.write(f"Starting new {run_name} training run\n")
    _file.write("-----------------------------------------------------------------\n")
    _run_name = run_name
    _slack_url = slack_url


def log(msg: str, end: str = "\n", slack: bool = False):
    """
    Logs a message to stdout and the log file.
    Optionally sends the message to Slack.
    """
    print(msg, end=end)
    if _file is not None:
        _file.write(f"[{datetime.now().strftime(_format)[:-3]}]  {msg}\n")
    if slack and _slack_url is not None:
        Thread(target=_send_slack, args=(msg,)).start()


def _close_logfile():
    """Closes the log file if it's open."""
    global _file
    if _file is not None:
        _file.close()
        _file = None


def _send_slack(msg: str):
    """Sends a message to the configured Slack webhook."""
    req = Request(_slack_url)
    req.add_header("Content-Type", "application/json")
    data = {
        "username": "tacotron",
        "icon_emoji": ":taco:",
        "text": f"*{_run_name}*: {msg}"
    }
    urlopen(req, json.dumps(data).encode())


atexit.register(_close_logfile)
