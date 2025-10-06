# CLIP-Challenge



PART 1:
And first, here are the 5 pictures which were given:

https://images.pexels.com/photos/36744/agriculture-arable-clouds-countryside.jpg
https://images.pexels.com/photos/825947/pexels-photo-825947.jpeg
https://images.pexels.com/photos/34044163/pexels-photo-34044163.jpeg
https://live.staticflickr.com/840/43380549381_004601c7ac_h.jpg
https://live.staticflickr.com/2404/2020522557_d1aa0a1066_k.jpg
(license:https://www.flickr.com/photos/briandewitt/2020522557/in/photolist-qCDWts-9LFMUT-uHeyNi-5G6LRz-45xGXP-hLkdkM-2ycbEf-3cHUMg-7y6Xh1-5Jqww1-hLm2hS-8gL5G-3WhbG4-EueMH-pfSn6-5dqjEz-4wHaBE-D3Ns2D-pBCiS-MpCHB-csWvU5-G97A4-6ZpBm5-6agE6f-5AGqd)

For each picture, I would like you to identify the text that best corresponds with the picture.

For this assignment part, I did one picture at a time :

For picture no. 1
What I observe: big sun close to horizon (sunrise/sunset), tree silhouetted on the right, birds flock, sky is orange.
Ideal single-word pick W (maximize): sunset (alt: sun, sunrise, sky, sunset is the most specific)
Formed caption reducing "A photo of a W": pick a word W that is semantically orthogonal — e.g. toaster → A photo of a toaster (predict low similarity)
Arbitrary caption downsize (C): a caption that depicts an entirely different scene, e.g. Close-up baby portrait with party hat.
Reasoning: CLIP will match prominent scene keywords (tree, sunset, sky). To enforce low similarity, select an ordinary object or scene with no common visual features (e.g. city street, toaster, spreadsheet) — the fewer common visual concepts, the lower the cosine.

For image 2.
What I see: studio photo of a dog (head-and-shoulders), plain background.
Best single-word W: dog (or more specifically: pitbull or staffordshire depending on breed; dog is safest)
Organized min "A photo of a W": A photo of a cat (or A photo of a sports car) — choose a clear object category
Random min C: Nighttime aerial photo of a city with neon lights. 



For image 3.
What I see: a foggy woods, leaves, upright tree trunks, and a wooden hay/feeding thing in the center.
Best single-word W: forest (alt: woods, trees)
Structured min "A photo of a W": A photo of a car or A photo of a skyscraper
Arbitrary min C: A close-up of a brightly colored fruit tart on a plate.

For image 4.
What I see: a cartoon drawing of a little white dog running eagerly inside, tongue sticking out, ears flapping, with motion lines indicating it is moving. The background is unadorned and simple.
Best single-word W: dog (alt: puppy, cartoon)
Structured min "A photo of a W": A photo of a car (or) A photo of a skyscraper
Arbitrary min C: A red sports car speeding along a racetrack.

For image 5.
What I see: a disheveled hotel room with two unmade beds, an ironing board in the middle, and subdued lighting from bedside lamps. The room is cozy but disordered.
Best single-word W: room (alt: hotel, bed)
Structured min "A photo of a W": A photo of a bird (or) A photo of a mountain
Arbitrary min C: A cartoon drawing of a running dog with floppy ears.
In brief, this is all that I did:
Step 1: Examining the Images
We had five images. To examine this manually and computationally:
Manual Observation: I carefully observed each image to determine its central object, scene, or action.

Example: One image was a sunset over fields, one was a portrait of a dog, and one was a forest shrouded in mist.


For the sunset photo: sunset, sun, sky, tree.

For the dog picture: dog, puppy, animal.

Candidate Captions: I generated "structured captions" such as A photo of a W and random captions that have no visual relation, e.g., A close-up of a smiling baby wearing a party hat.

Step 2: Using Code to Compute Similarity
To find the top and bottom matches, I applied the CLIP model using Python and HuggingFace transformers. The procedures were:


Load CLIP model and processor.

Download the images and process them to embeddings.

Process a list of candidate words/captions to embeddings.

Compute cosine similarity between each image embedding and each text embedding.

Identify:

The word W with maximum similarity (best caption for the image).

The structured caption with single word W with minimum similarity (worst caption for a template).

An arbitrary caption C with minimum similarity (entirely unrelated caption).

Step 3: Manual Search Insights
Manual experiments assisted me in getting a sense of CLIP's behavior:
For the dog photo, I tested animal → moderate similarity, dog → higher similarity, and puppy → slightly less than dog. This revealed the importance of specificity.

For the sunset photo, tree yielded moderate similarity, but sunset yielded the most similarity. Reducing captions was most effective when the caption represented a visually distinct scene, i.e., "toaster" for a landscape photo or "city at night" for a forest.

Step 4: Observations
CLIP has a strong preference for literal and exact matches between text and image.
Minor alterations in word usage can make a difference in perceived similarity.

Arbitrary captions with unrelated objects always yield low similarity, showing that CLIP can disentangle unrelated modalities.

Part 2: Maximizing Similarity for a Regular Image-Caption Pair
Part 2 prompted me to identify a common image and caption (not blank or artificial) that yields the highest cosine similarity.

Step 1: Selecting Candidate Images and Captions
I employed publicly accessible Pexels images with plain, explicit content:
Red apple on white background
White cat close-up

Pepperoni pizza
Mountain landscape
Person with a laptop


Step 2: Computing Similarity
With the same CLIP model, I calculated embeddings for all images and all captions, and then computed cosine similarity between all image-caption pairs.
similarity = cosine(CLIP(image), CLIP(caption))

I chose the pair with the top similarity score.

Step 3: Results
The most similar was usually obtained by:
Image: Red apple

Caption: "A photo of a red apple on a white background."

Reason: The image was simple, centered, and matched the caption literally. CLIP's embeddings are designed to match exact object descriptions to images.

Other very high-scoring pairs were the white cat and the pizza, but they were a little lower because they were a bit more visually complex or had several objects.

Step 4: Observations
CLIP prefers images with large, dominant objects and captions that precisely describe the image.

This exercise illustrated how CLIP spans visual and textual knowledge when both parties have a specific semantic notion.

Even without creating fake images, CLIP can detect pairs with extremely high similarity in real-world databases.

Step 5: Tools and Code
I ran everything inside a Google Colab notebook with the HuggingFace transformers library.

The notebook:
Loads CLIP
Loads images from URLs
Calculates embeddings
Brute-forces word/caption similarity for Part 1
Computes all image-caption similarities to identify the optimal pair for Part 2

This notebook may be reused with bigger word/caption lists for more extensive experiments.

✅ Key Takeaways
Part 1 enabled me to investigate how word selection and caption organization influence similarity.

Part 2 demonstrated the criticality of literal, descriptive exact matches to gain highest CLIP similarity.


Collectively, these components show CLIP closes the vision and language gap and why cosine similarity can be used to measure modal alignment.


