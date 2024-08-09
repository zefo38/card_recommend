import streamlit as st
import yaml
import streamlit_authenticator as stauth
import streamlit as st
# from langchain.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
import time
import pandas as pd


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

# authentication_status : 인증 상태 (실패=>False, 값없음=>None, 성공=>True)
if authentication_status == False:
    st.error("Username/password is incorrect")

# if authentication_status == None:
#     st.warning("Please enter your username and password")

if authentication_status:
    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f"Welcome {name}님")

        ## 로그인 이후


    st.title('💬 KB Chatbot')
    st.caption("더 나은 금융생활을 위한 맞춤형 서비스를 지원해드립니다.")
    st.header('')

    tab1, tab2= st.tabs(['가계부 챗봇' , '카드 추천'])
    with tab1:
        st.write('TAB1')
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for content in st.session_state.chat_history:

            with st.chat_message(content['role']):
                st.markdown(content['message']) # st.write

        if prompt := st.chat_input("메시지를 입력하세요."):
            with st.chat_message("user"):
                st.write(prompt)
                st.session_state.chat_history.append({"role":"user","message":prompt})

            with st.chat_message("ai"):
                response = f'{prompt}에 대한 답변입니다.'
                st.markdown(response)
                st.session_state.chat_history.append({"role":"ai","message":response})




# 카드 추천

# 데이터 불러오기



    with tab2:
        def data():

            df_transactions = pd.read_csv('card_transaction2.csv')
            card_benefit_df = pd.read_csv('card2.csv')
            df_transactions['날짜'] = pd.to_datetime(df_transactions['날짜'])

            df_transactions = df_transactions.loc[df_transactions['고객번호']==40]

            card_benefit = card_benefit_df.groupby(['카드명','혜택 카테고리']).mean().reset_index()

            card_benefit.drop(card_benefit.loc[card_benefit['혜택 카테고리'].isin(['음식점','편의점'])].index,inplace=True)


            # 카테고리별 이용 횟수를 계산하기 위한 데이터프레임 생성
            df_counts = pd.DataFrame(columns=['고객번호', '카드명', '혜택 카테고리', '이용 횟수','CREDIT분류'])
            

            # 각 고객에 대해 혜택 카테고리별 이용 횟수 계산
            rows_to_add = []
            for idx, row in df_transactions.iterrows():
                고객번호 = row['고객번호']
                for benefit_idx, benefit_row in card_benefit.iterrows():
                    카드명 = benefit_row['카드명']
                    혜택_카테고리 = benefit_row['혜택 카테고리']
                    신용카드분류 = benefit_row['Credit']
                    if row.get(혜택_카테고리, 0) > 0:
                        rows_to_add.append({
                            '고객번호': 고객번호,
                            '카드명': 카드명,
                            'CREDIT분류': 신용카드분류,
                            '혜택 카테고리': 혜택_카테고리,
                            '이용 횟수': 1
                        })
            

            # DataFrame으로 변환 후 concat

            df_counts = pd.concat([df_counts, pd.DataFrame(rows_to_add)], ignore_index=True)


            # 고객별 카드 혜택별 이용 횟수
            grouped_df = df_counts.groupby(['고객번호','카드명','혜택 카테고리','CREDIT분류']).sum().reset_index()
            grouped_df.columns = ['고객번호', '카드명', '혜택 카테고리','CREDIT분류', '이용 횟수']

            grouped_df['이용 횟수'] = pd.to_numeric(grouped_df['이용 횟수'])

            # 피벗 테이블 생성
            pivot_df = grouped_df.pivot_table(index=['고객번호','카드명','CREDIT분류'],columns='혜택 카테고리',values='이용 횟수',fill_value=0)

            # 컬럼 이름 정렬 (카테고리 순서 유지)
            pivot_df = pivot_df.sort_index(axis=1)

            row_sums = pivot_df.sum(axis=1)

            # 총합을 데이터프레임으로 변환
            row_sums_df = row_sums.reset_index(name='총합')

            # 고객번호와 총합으로 정렬
            sorted_row_sums = row_sums_df.sort_values(by=['고객번호', '총합'], ascending=[False, False])

            sum_df = sorted_row_sums.sort_values(['고객번호','총합'],ascending=[True,False])

            # 카드명 리스트를 생성
            card_list = ['다담_생활', '다담_직장인', '다담_쇼핑', '다담_레저', '다담_교육']

            # 반복문을 사용하여 동일한 작업을 처리
            for card in card_list:
                array_1 = (
                    sum_df.loc[sum_df['카드명'] == '다담']
                    .groupby(['고객번호', '카드명', 'CREDIT분류'])
                    .sum()['총합'].values
                    + sum_df.loc[sum_df['카드명'] == card]
                    .groupby(['고객번호', '카드명', 'CREDIT분류'])
                    .sum()['총합'].values
                )
                
                dd1 = array_1.tolist()
                sum_df.loc[sum_df['카드명'] == card, '총합'] = dd1
            sum_df = sum_df.loc[sum_df['카드명'] != '다담']

            cards1 = sum_df.sort_values('총합',ascending=False).head(3)['카드명'].tolist()


            st.write(cards1[0])

        st.subheader(f'{name}님에게 적합한 카드 추천')
        
        if __name__ == '__main__':
            data()

        if st.button('Say hello'):
            st.write('Why hello there')
        else:
            st.write('Goodbye')  




    # number = st.sidebar.slider('KB', 0, 10, 5)

