### Dockerization of Flask app######
 1. Ensure Requirement.txt has  all dependancies listed:
 2. create/ update  dockerfile
 3. for FLASK app, use CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"] # This line specifically instructs 
      # Docker to run our Flask app as a module, as indicated by the "-m" tag. 
      # Then it instructs Docker to make the container available externally, 
      # such as from our browser, rather than just from within the container. We pass the host port
 4. create docker image
   a) change directory to the project directory where app and requirement and dockerfile are kept
   b) docker build -t py-helloworld . # creates docker image. . stands for current directory
   c) docker images # list images
5. run and test docker image
   a) docker run -d -p6111:5000 py-helloworld
   b) docker ps # get all container instances
   c) test the application
   d) docker  stop  11392b92fd64 # gracefully kill container
6. tag the container : docker tag py-helloworld roshg/py-helloworld:v1.0.0
    docker images
7. # push the image
     a)docker push roshg/py-helloworld:v1.0.0
     b) login to docker with loginname and password
         docker login -u "myusername" -p "mypassword" docker.io
    c) docker push roshg/py-helloworld:v1.0.0
8. remove usused resources .
    docker container prune
    
    # docker ps # list all processes
    # docker stop container id # gracefully stops container

#################################################
##### ---Kubernetes -- ###########################

### Step  Deploy to Kubernetes 
This step is relevant for the "Exercise: Deploy Your First Kubernetes Cluster". 
The commands below are supposed to be run *only* after you have run the commands above. 
```
 PreRequiisite:
 1. Install Virtualbox
 2. install vagrant
 3. prepare Vagrantfile
      ensure right image file name in vagrantcloud is mentioned
       I had to comment out  config version no in vegrant file to make it work.
4. vagrant up
5. vagrant status
  
# Shortcut method to run the application with headless pods
kubectl run test --image roshg/py-helloworld:v1.0.0
# Another way to deploy the application
kubectl create deploy py-helloworld --image=roshg/py-helloworld:v1.0.0

# Display the pod name
kubectl get pods
# Copy the pod name from the output above
# Access the application on the local host
kubectl port-forward pod/py-helloworld-fcd468f98-rsj7p 6111:6111
```
Access the application in the local machine onhttp://127.0.0.1:6111/ 
