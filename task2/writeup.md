1 Wh is the role of the MLP in each block? Attention mixes information across positions. What does the MLP do that attention cannot? 

Attention is how tokens "talk" to each other; it mixes information across the sequence so a word can gather context from the words around it. However, attention doesn't actually compute or transform the meaning of that information deeplyThe MLP operates on each token individually. Once a token has gathered context from its neighbors via attention, the MLP acts as the "brain" for that specific position, applying non-linear transformations to process that context, memorize facts, and figure out what features to pass on to the next layer


2 Pre-norm versus post-norm: which did you use, and why is pre-norm easier to train for deep networks? 

Pre-norm is much easier to train for deep networks. The residual connection acts as an uninterrupted "highway" from the first layer straight to the last. Post-norm puts the LayerNorm directly inside that highway, which acts like a toll booth that distorts and slows down the gradients as they try to flow backward during training


3 Paste 300 characters of generated text. Compare qualitatively to Task 1's output

Compared to the bigram model in Task 1, this output is a massive leap forward. The Task 1 model only looked at one previous letter, so it produced semi-random gibberish that barely formed syllables the output is 
Mistan, plove,
That is have to feich that lust
thou and entimes bringedle, fair hape own this meld.

KATHARINA:
I fleet you, gongless all be canspowing
Bated me they not shores eyeye, 'tis hat, some.

ANTONIO:
Where or liberry to: wills, I am reven to so not beat,
noo than a su scince:
As fellh consest Kathariar, some felloked
ond twenty the rumps.

LUCENTIO:
The ming gentlemblening you it will not: weat switheys tousit! Laludgar is is shall so?

PETRUCHIO:
And thy cheem sute?

KATHARINA:
Anry g


4 From your ablation: which was more catastrophic, removing residuals or removing LayerNorm? Explain in terms of gradient flow what each removal breaks. 

Wthout residuals: The network suffers from "vanishing gradients." During backpropagation, the gradient has to be mathematically multiplied through every single attention and MLP matrix sequentially. Because these numbers are small, the gradient quickly shrinks to zero before it can reach the early layers, and the model completely stops learning 
Without LayerNorm: The gradients can still travel safely down the residual highway, but the scale of the numbers becomes wildly unstable. The loss jumps around chaotically because the activations aren't kept in check, resulting in poor learning—but it is still better than the total failure caused by removing residuals


5 What did you find hardest in this task? What clicked unexpectedly? 

Deriving the attention gradients by hand in Part C was the hardest part Keeping track of the matrix dimensions, indices, and the calculus chain rule for the Softmax Jacobian was mentally exhausting and required a lot of trial and error.
What clicked unexpectedly: The overall structure of the Transformer block. Seeing how the residual highway, LayerNorm, Multi-Head Attention, and MLP snap together like simple Lego bricks took the "magic" out of Large Language Model
