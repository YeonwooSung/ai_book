# CTR (Click Through Rate Prediction)

CTR prediction is to predict the probability of the user u clicking on the item v. The major difference
with collaborative filtering (Rendle et al., 2012) is that CTR prediction utilizes additional side
information about users and items as input features containing highly sparse categorical features.
Factorization Machines (FM) (Rendle, 2010) is the most representative CTR prediction model which
considers the first and second-order feature interactions from input features, simultaneously. Recently,
various CTR prediction models have been proposed to capture the high-order feature interactions via
deep neural networks Cheng et al. (2016); He & Chua (2017); Guo et al. (2017); Lian et al. (2018);
Qu et al. (2018); Song et al. (2019); Cheng et al. (2020). However, these CTR prediction models are
too over-parameterized to be deployed under limited computing resource scenarios.

## Gradient Boosting for CTR

[ et. al. [1]] stated that the Gradient Boosting models could overwhelm other SOTA models of CTR task by using the latest categorical feature encoding method.

LE (Label Encoding) converts a categorical feature into an arbitrary number, so that it would show sub-optimal performance since there are few correlations between the target category and its encoded number.
Meanwhile, TE (Target Encoding) changes the categorical feature into an informative number by calculating the mean of target values with each categorical feature (Micci-Barreca, 2001 [2]). However, TE causes overfitting by giving excessive information on each categorical feature (Schifferer et al., 2020 [3]).

The traditional categorical feature encoding methods, such as LE and TE, works fine, however, by using the latest categorical feature encoding method, we could overwhelm other SOTA algorithms.

### The latest categorical feature encoding methods

- LightGBM proposes the method that finds the optimal split of the categorical feature by using the Gradient and Hessian
- CatBoost suggests the modified version of Target Encoding, which uses the random shuffle to prevent the overfitting
- Also, there is a method that combines the XGBoost and K-Fold Target Encoding, which uses the K-fold CV to overcome the TE's overfitting issue

### Korean explanation

Gradient Boosting 으로 기존 Click-Through Rate(CTR) Prediction Task 의 SOTA 알고리즘들을 압도하는 성능을 낼 수 있다. (CTR Prediction 모델은 광고/추천 등 다양한 분야에서 활용되고 있습니다.)
다만, Gradient Boosting으로 기존 파라미터 기반의 SOTA 모델들을 압도하기 위해서는 Gradient Boosting을 단순히 적용하기만 해서는 안되고, 최신 categorical feature encoding method를 사용해야 한다.

최신 Categorical Feature Encoding 방법:

- LightGBM은 Gradient와 Hessian을 활용해 categorical feature의 optimal split을 찾는 방식을 제안했다.
- CatBoost에서는 Target Encoding의 오버피팅 현상을 막기 위해 random shuffle등을 통한 변형된 형태의 Target Encoding을 제안했다.
- K-fold cross validation을 통해 Target Encoding의 오버피팅을 해결한 K-Fold Target Encoding과 XGBoost가 조합된 방식이 제안되기도 했다.

더 자세한 내용은 [논문 저자의 블로그](https://hyperconnect.github.io/2021/04/26/hakuna-recsys-gb.html?fbclid=IwAR3P2SZKbdBhzUGBoKihERcLThqfaFSht1sTInW9zcircATs4AimhStQyFQ) 참조.

## References

[1] Joonyoung Yi and Buru Chang. [EFFICIENT CLICK-THROUGH RATE PREDICTION FOR DEVELOPING COUNTRIES VIA TABULAR LEARNING](https://arxiv.org/abs/2104.07553)

[2] Daniele Micci-Barreca. [A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems](https://dl.acm.org/doi/10.1145/507533.507538). ACM SIGKDD Explorations Newsletter, 3(1 :27–32, 2001.

[3] Benedikt Schifferer, Gilberto Titericz, Chris Deotte, Christof Henkel, Kazuki Onodera, Jiwei Liu, Bojan Tunguz, Even Oldridge, Gabriel De Souza Pereira Moreira, and Ahmet Erdem. [Gpu accelerated feature engineering and training for recommender systems](https://dl.acm.org/doi/10.1145/3415959.3415996). In Proceedings of the Recommender Systems Challenge 2020, pp. 16–23. 2020.
