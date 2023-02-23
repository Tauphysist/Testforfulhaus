# Test for Fulhaus

The current folder consists of several files, including:

1) Training file (train.py)

2) TensorFlow models saved in different formats (most of them start with model.*) 

3) API server (app.py)

4) Example of the Client request to the API (Client example.py)

5) Misc files such as Dockerfile and Yaml instructions 

Docker Container contains a full-scale API that can be deployed to any supporting platform. With the help of Github, Action docker containers are stored on my Dockerhub that could be easily retrieved (https://hub.docker.com/repository/docker/tauphysisit/testforfulhaus).

Tensorflow models are stored in different formats that could be used in various environments.

API is deployed on an open source www.pythonanywhere.com on the following link  http://tauphys.pythonanywhere.com/ that has 2 calls an "empty" one will return 
greetings, and the /predict_api requires a user to provide a binary image of the furniture to test and outputs the Class of the table and the accuracy of the prediction. An example of such a request is illustrated in the Client Example. (There is a chance of a long response due to the limitations of the platform)


