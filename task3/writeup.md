7 Compare your CNN baseline and your ViT on CIFAR-10. Which got better validation accuracy? Why might that be — what advantages does each architecture have? 

CNNs have a built-in advantage called "inductive bias." Because they use sliding convolutional filters, they naturally understand that pixels close to each other form local shapes (like edges and textures) and that an object is the same no matter where it appears in the frame. A Vision Transformer, however, starts completely blank. It has to learn the very concept of 2D geometry and spatial relationships from scratch. ViTs are incredibly powerful, but they typically require massive datasets (like ImageNet) to out-perform the natural, structural advantages of a CNN.


8 In your own words, explain why patching is necessary for ViT. Why not feed pixels directly? 

A tiny CIFAR-10 image is 32x32 pixels, which equals 1,024 total pixels. If we fed pixels directly into the model as individual tokens, the attention matrix would be 1,024 x 1,024—requiring over a million calculations per head, per layer. By grouping the pixels into 4x4 patches, we shrink the sequence length down to just 64 tokens. Patching makes the math physically possible to run on standard hardware without running out of memory


9 Explain the role of the CLS token. Why does the classifier read from CLS rather than averaging over patch tokens? 

As it passes through the transformer layers, the CLS token acts like a sponge. It uses attention to "look" at all the other image patches and dynamically gathers the most important global context into one single vector. We read from the CLS token because it has explicitly learned how to weigh the importance of different patches

10  In Task 2 you used a causal mask. In Task 3 (ViT) you removed it. Explain why this difference makes sense for the two tasks. What would happen to your ViT if you accidentally kept the causal mask? 

in task 2 text is generated sequentially.You have ti mask the future tokens so the model cannot cheat whereas for task 3 n image is static and completely visible all at once. To understand a patch containing a dog's ear, the model needs to be able to look across the image at the patch containing the dog's tail
if we keep the mask in ViT---It would destroy the model's spatial awareness. The first patch (top-left) wouldn't be allowed to see the rest of the image, and the last patch (bottom-right) could only look backward. The model would be trying to classify the image while looking through a progressively opening peephole, making it nearly impossible to understand the whole picture


11 Position embeddings for text encode token order. What do position embeddings for image patches encode, and why does the model need them? 

The transformer attention mechanism inherently treats all input tokens as an unordered "bag of words." If we didn't add position embeddings, the model would see the image as a scattered, randomized pile of puzzle pieces.The position embeddings act as the blueprint that tells the model exactly how to put the puzzle back together so it can understand the geometry of the image


12 What did you find hardest? What clicked unexpectedly? 

onverting a 2D image matrix into a flattened 1D sequence of patches while keeping all the batch, channel, and embedding dimensions perfectly aligned was challenging, especially when dealing with PyTorch out-of-memory crashes.The absolute simplicity of the Vision Transformer architecture. The "aha" moment was realizing that a ViT isn't some brand-new, complex computer vision algorithm. It is literally just the exact same text transformer from Task 2, but instead of feeding it a sequence of English words, we just fed it a sequence of pixel squares