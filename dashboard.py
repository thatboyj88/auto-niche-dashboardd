# dashboard.py
import streamlit as st
import json
from utils.automation_api import trigger_automation
from utils.telegram_notify import send_telegram_message

with open('config/settings.json','r') as f:
    settings=json.load(f)

TELEGRAM_BOT_TOKEN=settings['telegram']['bot_token']
TELEGRAM_CHAT_ID=settings['telegram']['chat_id']

st.set_page_config(page_title="AutoNiche Dashboard", layout="wide")
st.title("📊 AutoNiche Mobile Dashboard")

menu=st.sidebar.radio("Navigation",["Overview","Automation","Alerts"])

if menu=="Overview":
    st.header("Revenue Overview")
    st.metric("Weekly Revenue","$125.34")
    st.metric("Monthly Revenue","$532.10")
    st.header("Posts Overview")
    st.metric("Total Posts","58")
    st.metric("Posts Today","5")
    st.header("Traffic Overview")
    st.metric("Pageviews","1,234")
    st.metric("Affiliate Clicks","86")

elif menu=="Automation":
    st.header("Manual Automation Controls")
    if st.button("Run Scrape & Generate Posts"):
        with st.spinner("Running automation..."):
            result=trigger_automation()
            st.success(result)
            send_telegram_message(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,f"Automation Triggered: {result}")
    if st.button("Run Publish Only"):
        with st.spinner("Publishing..."):
            result=trigger_automation(mode='publish')
            st.success(result)
            send_telegram_message(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,f"Publish Triggered: {result}")

elif menu=="Alerts":
    st.header("Recent Alerts")
    for a in ["3 posts published","1 scraping error","Telegram notifications sent"]:
        st.write(a)

st.markdown("---")
st.markdown("Powered by AutoNiche AI")
