# Youtube Audio Data Processing Classification and Recommendation System
  
### INTRODUCTION
It is interesting to think that we the humans, have this pair of weirdly shaped organs called ‘ears’, which were initially evolved to detect the pressure and the mechanical waves caused by the vibrations in our environment, like the thin sound of vocal cords of a venomous snake or the noise of a huge tree collapsing. But as the human consciousness developed and we got comfortable with our environment, the concept of music came about, just like languages. Different cultures around the world developed their own sounds, their own instruments. Their sounds evolved alongside their cultures, and those cultures interacted with other cultures.
    
We are now living in the 21th century, where we can appreciate all kinds of sound created by any culture as a digital media, hundreds of years worth human created music inside our palms. This much data needs categorization. We the humans, are accustomed to labeling music genres and can distinguish the cultural differences of the sounds. The artistic nature of music means that the classification is going to be subjective and our genre picks are going to overlap. To categorize the already digitalized audio data we can use our computers, but the problem with our computers is that they have not evolved ears and consciousness, for them to understand music, we need to provide some mathematical methods. 

As mentioned, sound is a vibration, and vibrations can be expressed as mathematical signals. It wasn’t until the times of Pythagoras (569-475 BC) that the concept of harmonics was explored mathematically. Briefly, he discovered that some specific ratios of string lengths create pleasing combinations and for this reason he is considered to be the “father of harmony”. The discovery of these resonances are seen as the foundation of the western-hemisphere music composition. There were many other great mathematicians who contributed to the music theory like Euler. Alongside the musicians, the contributions of the mathematicians to music should not be disregarded.

In this project, we are going to deal with time and frequency domain features of audio signals using the equations and algorithms created by these mathematicians, then we are going to reduce those features into a couple of statistically valid numbers. The extracted features can also be used to classify other signal related tasks, for example in medical diagnosis of cardiac arrhythmia.

To classify audio signals, we are going to use Machine Learning methods. One interesting application related to our Project was done by Spotify. Spotify provides song features like mood, context, or the popularity of a song with their API. These features are used by many other students or programmers like us. Instead of using data that was already prepared for us, we wanted to create our own dataset and apply machine learning to it ourselves. 
![alt text](https://github.com/MerttErturkk/Youtube-Audio-Data-Processing-Classification-and-Recommendation-System/blob/main/FLOWCHART.jpg?raw=true)


We have built a Framework using Python. It uses YouTube as the data source as any kind of audio related media can be found there. The flowchart above demonstrates the core functionalities of the framework. We have built 3 separate programmes that takes a video URL as an input. 
1)	Programme that uses webscraping methods and filters the data then returns music related keywords
2)	Programme that applies a trained Machine Learning model to the extracted features and returns its predictions.
3)	Programme that returns 5 of the audio files that resembles the input file the most. 

Alongside our Framework we have built two machine learning models ready for prediction, first one was trained with GTZAN dataset for western music classification. The second model was trained with the Turkish Song Genres dataset we created using our automation method.

We wanted to build an alternative recommendation system for Youtube as we believe it can be improved with an automatic classification. Turkish Music video tags uploaded by common people do not usually contain music genre related tags, and many songs cannot be found unless specifically searched by its name or had already been consumed by a group of people to create a recommendation algorithm. 

In the greater scheme, a similar system to ours can be used for the discovery of those songs using automatically generated music genre tags. For this reason, we gave our project the title ‘Turkish Song Genre Classification’.





## EXTRA MODULES

beautifulsoup4            4.10.0
bs4                       0.0.1
ipython                   8.0.1
librosa                   0.8.1
matplotlib                3.5.1
nest-asyncio              1.5.4
numpy                     1.21.5
pandas                    1.4.0
plotly                    5.10.0
pydub                     0.25.1
requests                  2.27.1
requests-html             0.10.0
scikit-learn              1.0.2
scipy                     1.7.3
seaborn                   0.11.2
xgboost                   1.6.1
yt-dlp                    2022.3.8.2

