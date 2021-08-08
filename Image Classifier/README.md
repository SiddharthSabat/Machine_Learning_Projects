## Create Your Own Image Classifier

This project is part of my Udacity Machine Learning Intro with PyTorch Nanodegree program. The goal of the project is to develop an image classifier using PyTorch to recognize different species of flowers, simulataneously to create a command line application (i.e. an AI application) that can be trained on any set of labelled images. Here the trained network will be learning about flowers and end up as a command line application to predict new images using that trained model.  

### Install

This project requires **Python 3.7** and the following Python libraries :

- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included.

### Project Directory Content:

* `Image Classifier Project.ipynb`: This is the Jupyter notebook contains the main logics for training and predicting a flower class code for the project.
* cat_to_name.json: The Json file contains the a dictionary of the name of the flowers. 
* Data (flowers directory): The data used specifically for this assignemnt are a flower database. Inside flowers folder there are three subdirectories named train, test and validate folders available. These folders again containing a specific number of sub directories which corresponds to a specific category, clarified in the json file. e.g. if there is an image, say 'abc.jpg' is a 'Orchid' flower in the path '/test/5/abc.jpg' and the json file should have the label entry for that flowe as {...5:"rose",...}. The network to generalize better, it is advised to create the image database with bunch of images at least more than 10 with different angle, size and lighting conditions. This application can be used for any of the image classifer if the image database and the json file can be created in this manner. The image database used in this project can be found [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
* **Files for Command Line Arguments:**
    - train.py, predict.py, utils.py: These files contains code to run the command line version of the image classifier.  :
* readme.md: Readme file containing the details and instruction to run the application.

### Commands to run the Command line application
This project also can be run as a command line argument. Below are the instructions need to follow to run the application

* Train a new network on a data set with `train.py` file
    - Basic Command : `python train.py` (This will assume the the file path as the current directory for saving the checkpoint.pth file and train the model with default parameter values)
    - To train using the full argument list, below parameters to be passed in the command line. A Sample command with full arguments to run from project root directory as follows:
     - `python train.py --data_dir './' --arch 'densenet121' --learning_rate 0.001 --hidden_units 120 --epochs 1 --gpu 'gpu' --save_dir './checkpoint.pth'`
        - data direcotry:  --data_dir './' 
        - or save directory to save checkpoints: --save_dir './'
        - Arcitecture Choices (densenet121 or vgg16): --arch "vgg16" or --arch "densenet121"
        - Set hyperparameters: **--learning_rate 0.001** and **--hidden_layer1** 512 and **--epochs 3**
        - Use GPU or CPU for training:  --gpu gpu or --gpu cpu

* Predict flower name from an image with `predict.py` along with the probability of that name. Sample command and its arguments to pass are mentioned below.
    - Basic Command: Assuming the the model is trained with the checkpoint name as checkpoint.pth and saved in the project root directory, then the command with deafult arguments can be passed to predict the flower as below:
    * `python predict.py checkpoint.pth`. However, below is the command if the arguments need to be changed to other than default. 
        - python predict.py --filepath 'flowers/test/13/image_05769.jpg' --top_k 5 --category_names 'cat_to_name.json' --gpu 'GPU'  checkpoint.pth

    - Options:
        - Return top K most likely classes: --top_k 5
        - Use a dictionary mapping of categories to real names from the json file:  --category_names cat_To_name.json
        - Use GPU for inference:  --gpu 'gpu'

### GPU
Since the model is using deep convolutional neural network for training process, hence it is most recommended to train the model using GPU enabled laptops or any of the cloud provider like Google Colab, AWS, Microsoft Azure having GPU enabled instances.

### Hyperparameters
* To improve the accuracy of the network, hyperparameters such as **learning_rate**, **hidden_layer** and **--epochs** are used during the training process. Howoever in order to get a better prediction with the model generalizing better to testing set, these parameters tuning should be done carefully.

* By increasing the number of epochs the accuracy of the network on the training set gets better and better.Howevere if a large number of epochs are chosen, the network won't generalize well, and will result in high accuracy on the training image and low accuracy on the test images. 
* A big learning rate guarantees that the network will converge fast to a small error but it will constantly overshot
* A small learning rate guarantees that the network will reach greater accuracies but the learning process will take longer
* Densenet121 works best for images but the training process takes significantly longer than vgg16
* Default settings that are used in the project are lr=0.001, epochs= 3 and hidden_layer = 512. Test accuracy achived as 87% with densenet121 as my feature extraction model.

### Author
* [Sidharth Sabat](https://www.linkedin.com/in/sidharthsabat88/)

This project is licensed under the MIT License - see the LICENSE file for details