### 추천 시스템 개발 및 Streamlit 구현

(1) 사용 데이터 : 카드 결제 데이터 및 카드별 혜택 데이터

(2) 군집 분석 기반 추천 : 결제 패턴을 분석해 유사한 그룹에서 최적의 카드 추천

1.  사용 기술 : Pandas, Sklearn 
2.  차원 축소 방법 선정 : 비교군 3개 PCA, UMAP, PaCMAP
3. Clustering 방법 선정 : Kmeans, AgglomerativeClustering, GaussianMixture

(3) 콘텐츠 기반 추천 : 사용자 결제 데이터를 카드 혜택과 매칭하여 상위 3개의 카드를 추천

### (2) 군집 분석 기반 추천 : 결제 패턴을 분석해 유사한 그룹에서 최적의 카드 추천

1. **사용 기술** : Pandas, Sklearn을 사용하여 데이터 전처리 및 분석을 진행.
2. **차원 축소 방법 선정** : PCA, UMAP, PaCMAP 중 성능을 비교해 PCA를 선택하여 소비 카테고리의 차원 축소를 구현.
3. **Clustering 방법 선정** : K-means, AgglomerativeClustering, GaussianMixture 모델을 비교해 각 방법의 성능을 평가한 후, K-means를 최종적으로 선택하여 클러스터를 생성.

이 과정을 통해 유사한 소비 성향을 가진 그룹을 식별한 뒤, 각 그룹에 최적화된 카드 추천 시스템을 구축함.

---

### (3) 콘텐츠 기반 추천 : 사용자 결제 데이터를 카드 혜택과 매칭하여 상위 3개의 카드를 추천

사용자의 결제 데이터를 바탕으로 콘텐츠 기반 추천 알고리즘을 도입하여, 카드 혜택과 결제 내역을 매칭. 이를 통해 소비 패턴과 혜택이 가장 잘 맞는 상위 3개의 카드를 추천하는 기능을 개발함.

(2)-b. 상세 설명

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0eb912fc-ea25-4a36-9cbd-3afd4919b2ec/cb78390f-9969-4dc0-bd51-897371c20fb1/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0eb912fc-ea25-4a36-9cbd-3afd4919b2ec/c6a2f943-378e-4b5a-8976-849cee079c3b/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0eb912fc-ea25-4a36-9cbd-3afd4919b2ec/0e35a49d-c219-462c-a4e3-87447c7f6efa/image.png)

( 3개 중 2개로 선택지 축소 ) : 실루엣 점수

→ 차원 축소 후 실루엣 점수가 UMAP과 PaCMAP이 0.5점대로 유사하게 높았다. 

( 최종 UMAP 선정 ) :  카드 데이터 특성 고려

→ 카드 사용 패턴이 복잡한 경우, UMAP이 비선형 구조를 더 잘 보존할 수 있기 때문에 복잡한 데이터 구조 학습을 위해 UMAP 선정
