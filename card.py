import streamlit as st
import home
import yaml
import streamlit_authenticator as stauth
import time
import pandas as pd
import menu

# name, authentication_status, username = menu.main()
def card() :


# 로그인 창

    df = pd.read_csv('card2.csv')
    st.dataframe(df)
    
    st.subheader(f'{menu.main().name}적합한 카드 추천')

    if __name__ == '__main__':
         menu()

    if st.button('Say hello'):
        st.write('Why hello there')

    else:
        st.write('Goodbye')  
 

