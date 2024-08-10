import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import pacmap
from sklearn.metrics import silhouette_score
import streamlit as st

def cluster():
    """# 2. 데이터 확인 및 전처리"""

    df_transactions = pd.read_csv('card_transaction2.csv')

    credit = pd.read_csv('card2.csv')

    df_sum = df_transactions.drop('날짜', axis=1)
    df_sum = df_sum.groupby(['고객번호']).sum()

    df_sum2 = df_sum.drop(['부동산','여행','자동차','항공사','기기','기타'],axis=1)

    def detect_outliers(df, columns):
        for column in columns:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            boundary = 1.5 * iqr

            index1 = df[df[column] > q3 + boundary].index  # Q3+1.5∗IQR
            index2 = df[df[column] < q1 - boundary].index  # Q1−1.5∗IQR

            df[column] = df[column].drop(index1)
            df[column] = df[column].drop(index2)

        return df
    detect_outliers(df_sum2,['숙박','교통','카페','온라인결제','병원','화장품','주유소','동물병원','해외직구','편의점','스포츠','보험','해외결제','문화'])
    df_sum2=df_sum2.fillna(0)

    """# 3. 데이터 스케일링 - 표준화"""

    # 데이터 표준화
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaled_df = scaler.fit_transform(df_sum2)

    # PaCMAP 객체 생성
    embedding = pacmap.PaCMAP(
        n_components=2,  # 2차원으로 축소
        n_neighbors=None,  # 기본값은 None으로, 데이터에 따라 설정
        MN_ratio=0.5,  # MN 비율
        FP_ratio=2.0  # FP 비율
    )

    # 데이터 변환
    pac_df = embedding.fit_transform(scaled_df, init="pca")

    # K-means 클러스터링 적용
    kmeans = KMeans(n_clusters=3)  # 클러스터 수를 적절히 설정하세요
    clusters = kmeans.fit_predict(pac_df)

    weights = {
        '교통': 1.0,
        '온라인결제': 1.3,
        # '기기': 1.0,
        # '기타': 0.8,
        '독서실': 1.0,
        '병원': 1.1,
        '쇼핑': 1.3,
        '문화': 1.0,
        '음식점': 0.9,
        '서점': 1.2,
        '카페': 0.9,
        '여행': 1.0,
        '동물병원': 1.0,
        '영화': 1.0,
        '스포츠': 1.0,
        '미용실': 1.0,
        '편의점': 1.2,
        '학원': 1.0,
        '이동통신': 1.2,
        '세탁소': 1.2,
        # '항공사' : 1.0,
        '숙박': 1.0,
        '배달': 1.0,
        '해외결제': 1.0,
        '대형마트': 1.0,
        'OTT구독': 1.0,
        '해외직구': 1.0,
        '테마파크': 1.2
    }

    card_df1 = pd.DataFrame({
        'CardID': [10, 11, 12, 13, 14, 15, 16, 17],
        'CardName': ["MyWESH_먹는데진심", "MyWESH_노는데진심", "MyWESH_관리에진심", "다담_생활", "다담_교육", "다담_쇼핑", "다담_직장인", "다담_레저"],
        '쇼핑': [0, 1, 0, 1, 0, 1, 0, 1],
        '카페': [1, 0, 0, 0, 0, 0, 1, 0],
        '음식점': [1, 0, 0, 0, 0, 0, 0, 0],
        '숙박': [0, 0, 0, 0, 0, 0, 0, 0],
        '서점': [0, 0, 1, 0, 1, 0, 0, 0],
        '온라인결제': [0, 0, 0, 0, 0, 1, 0, 0],
        '교통': [0, 1, 0, 0, 0, 0, 1, 0],
        '미용실': [0, 0, 1, 1, 0, 0, 0, 0],
        '병원': [0, 0, 0, 0, 0, 0, 0, 1],
        '주유소': [0, 0, 0, 1, 1, 1, 1, 1],
        '화장품': [0, 1, 1, 1, 0, 0, 0, 0],
        '편의점': [0, 1, 1, 1, 0, 1, 1, 0],
        '주차장': [1, 0, 0, 0, 0, 0, 0, 0],
        '문화': [0, 1, 1, 1, 0, 0, 0, 1],
        '이동통신': [1, 1, 1, 1, 1, 1, 1, 0],
        '학원': [0, 0, 0, 0, 1, 0, 1, 0],
        '스포츠': [0, 0, 1, 0, 0, 0, 0, 1],
        '동물병원': [0, 0, 0, 0, 0, 0, 1, 0],
        '세탁소': [0, 0, 0, 0, 1, 0, 0, 0],
        '영화': [0, 1, 0, 0, 0, 0, 0, 0],
        # '부동산' : [0,0,0,0,0,0,0,0],
        # '자동차' : [0,0,0,0,0,0,0,0],
        '해외직구': [0, 0, 0, 0, 0, 0, 0, 1],
        '택시': [0, 1, 0, 0, 0, 0, 1, 0],
        '테마파크': [0, 0, 0, 1, 0, 0, 0, 0],
        # '항공사' : [0,0,0,0,0,0,1,0],
        '배달': [1, 0, 0, 0, 0, 0, 0, 0],
        '해외결제': [0, 0, 0, 1, 1, 1, 1, 1],
        '대형마트': [0, 0, 0, 1, 0, 0, 0, 0],
        'OTT구독': [1, 1, 1, 0, 0, 0, 0, 0]

    })


    card_df2 = pd.DataFrame({
        'CardID': [18, 19, 20, 21],
        'CardName': ["EasyAll티타늄_A", "EasyAll티타늄_B", "EasyAll티타늄_C", "EasyAll티타늄_D"],
        '쇼핑': [0, 0, 1, 1],
        '카페': [0, 0, 0, 1],
        '음식점': [1, 0, 0, 0],
        '숙박': [0, 0, 0, 0],
        '서점': [0, 0, 0, 0],
        '온라인결제': [0, 1, 0, 0],
        '교통': [0, 1, 0, 0],
        '미용실': [0, 0, 0, 1],
        '병원': [1, 0, 0, 0],
        '주유소': [0, 1, 0, 1],
        '화장품': [0, 0, 1, 0],
        '편의점': [0, 1, 0, 0],
        '주차장': [0, 0, 0, 0],
        '문화': [0, 0, 0, 1],
        '이동통신': [0, 1, 0, 1],
        '학원': [1, 0, 0, 0],
        '스포츠': [0, 0, 1, 0],
        '동물병원': [0, 0, 0, 0],
        '세탁소': [0, 0, 0, 0],
        '영화': [0, 0, 0, 1],
        # '부동산' : [0,0,0,0],
        # '자동차' : [0,0,0,0],
        '해외직구': [0, 0, 0, 0],
        '택시': [0, 0, 0, 1],
        '테마파크': [0, 0, 0, 0],
        # '항공사' : [0,0,0,0],
        '배달': [0, 0, 0, 1],
        '해외결제': [1, 0, 0, 0],
        '대형마트': [0, 1, 0, 0],
        'OTT구독': [0, 0, 0, 1]

    })

    card_df3 = pd.DataFrame({
        'CardID': [22, 23],
        'CardName': ['굿데이올림카드', 'BeVV카드'],
        '쇼핑': [0, 1],
        '카페': [0, 0],
        '음식점': [0, 0],
        '숙박': [0, 1],
        '서점': [0, 0],
        '온라인결제': [0, 0],
        '교통': [1, 0],
        '미용실': [0, 0],
        '병원': [1, 0],
        '주유소': [1, 0],
        '화장품': [0, 0],
        '편의점': [1, 0],
        '주차장': [0, 0],
        '문화': [0, 0],
        '이동통신': [1, 0],
        '학원': [1, 0],
        '스포츠': [0, 1],
        '동물병원': [0, 0],
        '세탁소': [0, 0],
        '영화': [0, 1],
        # '부동산' : [0,0],
        # '자동차' : [0,0],
        '해외직구': [1, 0],
        '택시': [0, 0],
        '테마파크': [0, 0],
        # '항공사' : [0,0],
        '배달': [0, 0],
        '해외결제': [1, 0],
        '대형마트': [1, 0],
        'OTT구독': [0, 0]

    })

    card_df = pd.concat([card_df1, card_df2, card_df3], axis=0)  # 열방향 연결, 데이터프레임

    """## 5.3 카드 추천  
    ### 5.3.1 카드 추천 점수 계산
    """

    # 점수 계산 함수

    df_sum2['cluster'] = kmeans.fit_predict(pac_df)

    cluster_means = df_sum2.groupby('cluster').mean()

    def weighted_sum_score(cluster_mean, card, weights):
        return sum(cluster_mean[feature] * card[feature] * weights.get(feature, 1) for feature in card.index[2:])

    # 중복 추천 함수 (상위 N개 카드 추천)
    def recommend_top_n_cards(cluster_mean, card_df, weights, n=3):

        recommendations = {}
        for cluster, mean_values in cluster_mean.iterrows():
            scores = []
            for _, card in card_df.iterrows():
                score = weighted_sum_score(mean_values, card, weights)

                scores.append((card['CardID'], card['CardName'], score))
            # 상위 N개 카드 추천
            top_n_cards = sorted(scores, key=lambda x: x[2], reverse=True)[:n]
            recommendations[cluster] = top_n_cards
        return recommendations

    # 각 클러스터에 대해 상위 3개 카드 추천
    top_n_recommendations = recommend_top_n_cards(cluster_means, card_df, weights, n=3)

    """# 6. 특정 사용자에게 카드 추천  
    ## 6.1. 사용자 소비패턴과 클러스터 유사도 확인
    """

    individual_card_df = df_sum2.loc[10]

    individual_df = individual_card_df.to_frame().T
    individual_mean = individual_df.drop('cluster', axis=1)


    from sklearn.metrics.pairwise import cosine_similarity

    # 클러스터 평균 데이터와 개인 소비 패턴 벡터화
    cluster_means_array = cluster_means.values
    individual_mean_array = individual_mean.values.reshape(1, -1)

    # 코사인 유사도 계산
    similarity_scores = cosine_similarity(cluster_means_array, individual_mean_array).flatten()

    # 가장 유사한 클러스터 찾기

    most_similar_cluster = np.argmax(similarity_scores)

    card3 = []
    for i in range(0, 3):
        card3.append(top_n_recommendations[most_similar_cluster][i][1])
    st.write(card3)