import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import get_keys, pretty_json
from utils.ipfs_upload import *
from utils.eth_write import *
from model.get_world_data import *

import requests
import logging
from telegram import Update, Bot
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
import schedule
import time
from datetime import datetime
# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
bot = None

def pipeline():
    filename = call_model()
    pretty_json(f"./../model/{filename}")
    hash = pinFile(f"./../utils/outputs/{filename}")

    timestamp = math.floor(datetime.datetime(
        datetime.now().day,
        datetime.now().month,
        datetime.now().year,
        0,0
     ).timestamp())
    print(f"Today's timestamp is {timestamp}")
    tx_hash = appendHash(timestamp, [[0,0]]*4, hash)

    msg = f"A new model prediction has been released on the website! Check the latest heatmap on https://microplastictracker.site\n"
    + f"Today's data was published with IPFS hash {hash} and Ethereum TX hash {tx_hash}.\n"

    bot.send_message(chat_id="@uplastics", text=msg)

schedule.every(24).hour.do(pipeline)

def main() -> None:
    tele_token = get_keys("./../keys/api_keys.json", ["TELEGRAM_TOKEN"])[0]
    print(tele_token)
    # updater = Updater(tele_token)
    global bot
    bot = Bot(tele_token)

    while True:
        schedule.run_pending()
        time.sleep(1)
        


if __name__ == "__main__":
    
    initPinata()
    initCreds()
    initKeys()
    initWeb3(f"https://goerli.infura.io/v3/{eth_keys[2]}", contract_address, contract_json)

    main()
    
