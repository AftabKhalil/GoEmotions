Qs1_and_Qs2.ipynb has the solution to question 1 and 2, i.e it measures the effect of changing the hyper parameters of the given function and then do the same task for two more models.
It also exports the model from question 1 to google drive from where it is downloaded to use in the Docker container required in Qs5.
Qs3.ipynb transform the data from multiple prediction classes to only 2 classes of 'positive' and 'negative', the model is exported to google drive and downloaded to use in qs5.
The folder Qs4_and_Qs5 contains the Docker container exposed via fastAPI to use the above exported models funcionalities.
A simple UI is also added to call FastAPI endpoints!

GitRepo https://github.com/AftabKhalil/GoEmotionsDetection