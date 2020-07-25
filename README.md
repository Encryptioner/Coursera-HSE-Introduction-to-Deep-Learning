# Introduction to Deep Learning

## Requirements

1. Python 3.6 

## Installation (For Ubuntu 18.04 - Virtual Environment)

1. Run `sudo apt-get install -y python3-pip`
2. Run `sudo pip3 install virtualenv` 
3. Run `virtualenv ENVIRONMENT_NAME`. You can Run it in your home/project directory. It will be then operated from there.
4. To active find `.virtualenvs` folder & run `source ENVIRONMENT_NAME/bin/activate`. You will get shell.
5. You also can run `workon ENVIRONMENT_NAME` to get the shell.
6. To install requirement packages `pip install -r requirements.txt`
7. To install packages run `pip3 install <some-package>`
8. To deactivate run `deactivate` on the shell.


## Others

Course can be found [here](https://www.coursera.org/learn/intro-to-deep-learning)

Notebook for quick search can be found [here](https://ssq.github.io/2017/11/19/Coursera%20HSE%20Advanced%20Machine%20Learning%20Specialization/)

- Week 1 Introduction to optimization
  - Train a linear model for classification or regression task using stochastic gradient descent
  - Tune SGD optimization using different techniques
  - Apply regularization to train better models
  - Use linear models for classification and regression tasks
  - [x] [Linear models and optimization](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%201%20PA%201%20Linear%20models%20and%20optimization)

- Week 2 Introduction to neural networks
  - Explain the mechanics of basic building blocks for neural networks
  - Apply backpropagation algorithm to train deep neural networks using automatic differentiation
  - Implement, train and test neural networks using TensorFlow and Keras
  - [x] [Going deeper with Tensorflow](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%202%20PA%201%20Going%20deeper%20with%20Tensorflow)
  - [x] [my1stNN](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%202%20PA%202%20My1stNN)
  - [x] [Getting deeper with Keras](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%202%20PA%203%20Keras%20task)
  - [ ] [Your very own neural network]()
  
- Week 3 Deep Learning for images
  - Define and train a CNN from scratch
  - Understand building blocks and training tricks of modern CNNs
  - Use pre-trained CNN to solve a new task
  - [x] [Your first CNN on CIFAR-10](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%203%20PA%201%20Your%20first%20CNN%20on%20CIFAR-10)
  - [x] [Week 3 PA 2 Fine-tuning InceptionV3 for flowers classification](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%203%20PA%202%20Fine-tuning%20InceptionV3%20for%20flowers%20classification)
  
- Week 4 Unsupervised representation learning
  - Understand what is unsupervised learning and how you can benifit from it
  - Implement and train deep autoencoders
  - Apply autoencoders for image retrieval and image morphing
  - Implement and train generative adversarial networks
  - Understand basics of unsupervised learning of word embeddings
  - [x] [Autoencoders](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%204%20PA%201%20Simple%20autoencoder)

- Week 5 Deep learning for sequences
  - Define and train an RNN from scratch
  - Understand modern architectures of RNNs: LSTM, GRU
  - Use RNNs for different types of tasks: sequential input, sequential output, sequential input and output
  - [x] [Generating names with RNNs](https://github.com/SSQ/Coursera-HSE-Introduction-to-Deep-Learning/tree/master/Week%205%20PA%201%20Generating%20names%20with%20RNNs)
  
- Week 6 Final Project
  - [ ] [Image Captioning Final Project]()
