# utils/telegram_notify.py
import requests

def send_telegram_message(bot_token, chat_id, message):
    try:
        url=f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload={'chat_id':chat_id,'text':message}
        r=requests.post(url,data=payload)
        return r.status_code==200
    except:
        return False
