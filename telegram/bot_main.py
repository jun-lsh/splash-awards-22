import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.utils import get_keys
from utils.ipfs_upload import *
from utils.eth_write import *

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )

import requests
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends explanation on how to use the bot."""
    await update.message.reply_text("Welcome to the Splash Microplastics Tracker channel! "
                                    + "This is merely a prototype and not production ready~")

def main() -> None:
    tele_token = get_keys("./../api_keys.json", ["TELEGRAM_TOKEN"])[0]
    application = Application.builder().token(tele_token).build()
    application.add_handler(CommandHandler(["start"], start))
    application.run_polling()

if __name__ == "__main__":
    main()
