# Multimodal Transformer - Final Writeup

## 1. Training Summary
* **Hardware Used:** CPU
* **Training Steps:** 500 (Kept short to avoid 20+ hours of CPU time)
* **Final Training Loss:** 3.79
* **Final Validation Loss:** 4.24

## 2. Testing the Model (Scores)
To see if the model actually learned anything, we tested it to see if it could match images to the correct text. If it was just guessing randomly out of 1,000 items, it would get a score of about **0.1%**.

Here are my model's scores:
* **Image-to-Text Score:** 0.2%
* **Text-to-Image Score:** 0.3%

Even though the training was cut very short, the model performed 2 to 3 times better than random guesswork. This proves the code works and the model is starting to learn!

## 3. Image Search Engine 
For the final test, I built a Text-to-Image search engine. I typed the sentence *"a dog running on the grass,"* and the model looked through a folder of images to find the top 5 closest matches. 

![Search Results](task6/search_results.png)

Because it only trained for 500 steps, it isn't perfect at finding specific details yet. However, it successfully learned to look for big clues like the color green (for grass) or general animal shapes. 

## 4. What I Learned
* **Windows requires special code:** When using PyTorch to load data quickly on Windows, you have to wrap your code in a `if __name__ == '__main__'` block, or else it crashes.
* **How CLIP works:** The model learns by looking at a picture and a sentence at the same time and mathematically pulling them closer together if they match.
* **Computer power matters:** Training AI on a normal laptop CPU is incredibly slow. To make it work in a reasonable amount of time, shrinking the images (like down to 64x64 pixels) is a huge help.