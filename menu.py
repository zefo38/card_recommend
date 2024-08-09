import streamlit as st
import home
import streamlit as st
import yaml
import streamlit_authenticator as stauth
import streamlit as st
# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
import time
import pandas as pd
import card




def main() :
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=stauth.SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    # 로그인 창
    name, authentication_status, username = authenticator.login("main")


    st.title('💬 KB Chatbot')
    st.caption("더 나은 금융생활을 위한 맞춤형 서비스를 지원해드립니다.")

    menu = ['Home', 'card', 'chatbot']
    choice = st.sidebar.selectbox('메뉴', menu)
 
    if choice == menu[0] :
        home.run_home()
    elif choice == menu[1]:
        card.card()

if __name__ == '__main__':
            main()