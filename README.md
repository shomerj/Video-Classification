<h2>Readme</h2>

<h2> Video Classification of Semi Natural Human Movements </h2>

<h3>Overview</h3>

The goal of this project is to build a video classifier that correctly predict five different human movements. The movement that I wish to classify are: throwing, jumping, punching, push-ups, and pull-ups,

<h3>Motivation</h3>

My interest in this problem came about when I was introduced to a new type of camera that has a built-in GPU. This got me thinking, could you possibly train a model, extract the weights/parameters, load them into the camera, and conduct real time video classifications? The neurons began to fire (pun intended) and the possible applications of this type of technology started to surface. With some conversation and direction from my peers we started to discuss the implementation of this in the domain of security. What if we could classify suspicious behavior? What does that mean to a computer? With these thoughts ingrained in my head I decided to take on a small subset of the larger problem stated previously. I found a video dataset from the University of Central Florida called UCF101. With the dataset at hand, I categorized the dataset into five human actions that I saw as a good segue into the broader problem presented: throwing, punching, jumping, push-up, pull-ups.


<h3>Data</h3>

The data I am using for this project is the [UCF101 - Action Recognition Data Set](http://crcv.ucf.edu/data/UCF101.php). The set contains around 13,000 videos with 101 categories. I will be using a subset of the categories to create my own 5 classifications. All the clips are from YouTube with a fixed frame rate of 25 FPS and a resolution of 320 X 240. I will be creating JPEG images from each frame and using the images over a predetermined timestamp to train my model.


<h3>Schedule</h3>

[Here is a link to my schedule](https://docs.google.com/spreadsheets/d/1ykt7rSHiWaSsG733xhWhU9j5NEZ2z1nzB0x6uOXK3p4/edit?usp=sharing)
