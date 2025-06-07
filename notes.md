generatively pre-trained transformer

gelu over relu
- because it has a smoother curve meaning it approximates better

EX1: The n-dimensional tensor mastery challenge: Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension (answer is in nanoGPT).

next steps: i. add RoPe for better position encoding, ii. the model is way too overgeneralized, fix that. 

i. rope added. train loss went down to 1.6082 but val loss came to 1.7799

regularizing the model
i. increasing dropout 0.0 -> 0.5 -> 0.2
ii. increasing model capacity from 64 -> 128