12. Why do we divide attention scores by sqrt(d_k)? Connect your answer to the variance of dot products and to the shape of the softmax. 
the attention score is computed as dot product between key and query vectors.As the magnitude of these vectors increases the magnitude of the dot vector also increases 
if these values become very large The softmax function generates very sharp probability distributions in case which one position receives very large attention while others do not...the gradients become very small dividing by sqrt(d_k) keeps the magnitude in control and stable so the model can learn more effectively



13. Why is the causal mask applied before softmax?
the casual mask prevents a token from looking at the future tokens here future positions are assigned a value of negative infinity defore applying softmax when a score is set to negative infinity before applying softmax it's probabilit becomes exactly zero and the further probabilities are reassigned correctly



14. Describe Q, K, and V in your own words.
a query represents what a current token is looking for a key represents the information available at the particular position the attention score is calculated by comparing the query with all keys if the query and the key are similar that position receives a higher attention weight 
The value contains the actual information that will be passed forward Once the model decides which positions are important it computes a weighted combination of their value vectors 



15. Why does the single-head attention model only slightly outperform the bigram model?
the improvement was relatively small because the model still has several limitations first the context length was only 8 tokens meaning the model could look only at a small portion of the previous text 
modern transformers use multiple heads so they can learn efficiently 
also the model did not consist of feed forward networks,residual connections and normalization 



16. Comparison of Generated Samples
bigram model sample 

MARI he avayokis erceller thour d, myono thishe me tord se by he me, Forder anen: at trselorinjulour t yoru thrd wo ththathy IUShe bavidelanoby man ond be jus as g e atot Meste hrle s, ppat t JLENCOLI



attention model sample
Y: waenche edrth,
NG ive weiteve.
Thave ndetrid wifrud has is.
 JUS:
Th
Misho om sod winom bulall kangesoue be yorime hacigu tha ph med mpeat oead out acarmitso sas
thentesein,
Y:
Tel sst forsheriy an


the biagram model was able to learn character freq but it's output was nonsensical on the other hand the attention model generated text that looked slightly more structured the output was still far from meaningful text but it showed was using context from previous tokens rather than relying only on current character