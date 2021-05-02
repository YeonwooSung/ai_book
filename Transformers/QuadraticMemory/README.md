# Quadratic memory issue

Since the Transformer Layers compute the attention map by doing the dot product with K and V (Key and Value), the size of the memory increases quadratically when increasing the size of the input data.

## Linformer

[Wang. et. al [1]](https://arxiv.org/abs/2006.04768) stated that the self-attention is low rank. Based on this theorem, they proposed a new self-attention mechanism, which reduces the overall self-attention complexity from O(n^2) to O(n) in both time and space. The resulting linear transformer, the Linformer, performs on par with standard Transformer models, while being much more memory- and time-efficient.

### 해설 (Linformer)

"low rank"란 저차원이라는 뜻으로, 이 논문의 저자들은 self-attention을 통해 만든 attention map을 보면 대부분의 정보가 낮은 차원 (eigenvalue index < 128)에 몰려 있다는 점을 밝혔고, 고차원의 데이터를 저차원에 투영(projection)했을 때, 투영된 데이터 점들 사이의 거리는 원래의 데이터 점들 사이의 거리와 큰 괴리가 없다는 점 역시 밝혔다. 이를 통해서, 논문의 저자들은 V와 K의 차원을 줄이는 Projection layer를 Attention mechanism 내에 추가함으로써 별다른 성능저하 없이 어텐션 맵의 차원을 줄이는 데 성공해냈다.

## Performer

## Longformer

## References

[1] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma. [Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)
