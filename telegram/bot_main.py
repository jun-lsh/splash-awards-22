import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import get_keys
from utils.ipfs_upload import *
from utils.eth_write import *

import requests
import logging
from telegram import Update, Bot
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

def main() -> None:
    tele_token = get_keys("./../keys/api_keys.json", ["TELEGRAM_TOKEN"])[0]
    print(tele_token)
    # updater = Updater(tele_token)
    bot = Bot(tele_token)
    


if __name__ == "__main__":
    main()
    
