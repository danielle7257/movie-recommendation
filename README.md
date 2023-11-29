# Introduction: 
The recommendation systems are an important problem to solve. For any major e-commerce platform, the major share of their revenue is generated from the recommendation systems. Netflix, YouTube, Amazon Prime, and many major streaming services improve their recommendation systems to generate revenue. 

This project movie recommendation systems is an attempt to solve the challenge. We have used a hybrid approach in solving this challenge. We have used both content-based filtering and collaborative filtering approaches to solve the recommendation systems. 

# How to run the project: 

1. Clone the project repo to your local folder. 

2. Download the below dataset files from the below link.

   ## Link: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
  
   ### Files to download:
   credits.csv, 
   keywords.csv, 
   links_small.csv, 
   movies_metadata.csv, 
   ratings_small.csv.

3. Place the above dataset files in the backend folder of your project repo. 

4. Open a terminal inside the backend folder and try to Install the below python libraries. 

   ## Libraries: 
   Use the below commands to install the libraries. 

   ```bash
   pip install Flask tmdbv3api requests pandas numpy matplotlib seaborn scipy nltk scikit-learn surprise fuzzywuzzy
   ```

5. In the main folder, open a terminal and run the below command to install the npm modules. 

   ```bash
   npm install
   ```
   If you do not have the npm installed before, you can refer to the below link and configure it to your system.

   ## Link: https://docs.npmjs.com/downloading-and-installing-node-js-and-npm

6. Once all the libraries are installed, open a terminal in the main project repo and run the below command to start the react server. 

   ```bash
   npm start
   ```
   Runs the app in the development mode.
   Open http://localhost:3000 to view it in your browser.
  
   The page will reload when you make changes.
   You may also see any lint errors in the console.

7. Now open another terminal inside the backend folder and run the main.py file using the below command.

   ```bash
   python main.py
	 ```
   This command will start the backend Flask server. It will take some time to run the application. You can open a browser and check the link http://127.0.0.1:5000

8. You can watch the demo at
   
   [![Watch the video](https://img.youtube.com/vi/3AXcZaP2aoo/hqdefault.jpg)](https://youtu.be/3AXcZaP2aoo)
