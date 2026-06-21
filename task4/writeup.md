 17 the exact difference between self-attention and cross-attention in one or two sentences. What changes mechanically, and what changes semantically? 

In self-attention, the Queries, Keys, and Values all come from the exact same input sequence (like a text sentence). In cross-attention, the Queries come from one source (the text), while the Keys and Values come from a completely different source whereas semantically Self-attention learns relationships within a single type of data (e.g., understanding how words in a sentence relate to each other). Cross-attention acts as a bridge between different types of data, allowing one modality



18  In your decoder block, you have three sub-layers: causal self-attention, cross-attention (not causal), and MLP. Explain why each one is or is not causally masked. What would go wrong if you swapped which were causal? 

Text generation is a step-by-step process. You must mask this layer so the model cannot "cheat" by looking at future words it hasn't generated yet.In cross attention the text needs to look at the entire image all at once to figure out what to say.MLP (Not Masked): The Feed-Forward network processes each token individually. It does not mix information across the sequence length, so there is nothing to mask.If self-attention was unmasked, the model would cheat during training and completely fail when trying to generate real text. If cross-attention was masked, the first word of the caption could only look at the first corner of the image, the second word at the first two corners, etc., which destroys the model's ability to understand the whole picture.


19  In cross-attention, the output sequence length equals the query length, not the context length. Why? Walk through the shapes. 

The length of the output is determined entirely by who is "asking the questions" (the Queries).if you have 10 text tokens asking questions about 50 image patches, you still only need 10 answers back.


20 If your vision encoder produced features with a different n_embd than your text decoder, how would you handle it? List two ways. 

If the Vision Transformer outputs a dimension size of 256, but the Text Decoder expects 128.
A Projection Layer: Add a standard Linear layer (nn.Linear(256, 128)) directly after the vision encoder to squash the image features down to match the text dimension before passing them into the cross-attention block
Custom Key/Value Matrices: Inside the cross-attention layer itself, you can set the Key and Value linear layers to accept the vision dimension. For example, self.key = nn.Linear(256, head_size) and self.query = nn.Linear(128, head_size).


21 Describe what you saw in your cross-attention visualization. Did the model attend to anything meaningful? Why or why not, given the simplicity of your synthetic dataset? 

Because I had to restrict my training loop to just 10 steps to accommodate my CPU hardware constraints, the resulting cross-attention heatmap looked like random noise and static. The model simply did not have enough time or data to learn meaningful alignments (like mapping the word "cat" to the center image patches).generating the map proved that the mathematical dimensions aligned perfectly and that the gradients flowed backwards smoothly from the text generation all the way through the image encoder


22 Connect what you built in Task 4 to one real multimodal model you've heard of (CLIP, Flamingo, BLIP, LLaVA, GPT-4V — any). In your own words, what is the same and what is different?

What I built in Task 4 is structurally very similar to DeepMind's Flamingo model.Just like my model, Flamingo uses a "late-fusion" approach. It keeps a pre-trained vision encoder and a pre-trained language model mostly separate, and inserts brand-new cross-attention layers into the text model so the text tokens can "look" at the image features.What is different: Aside from being billions of parameters larger, Flamingo uses a "Perceiver Resampler" module to compress thousands of image patches down into a few dozen dense visual tokens before passing them into the cross-attention layer. In my toy model, I just passed every raw image patch directly into the cross-attention block.