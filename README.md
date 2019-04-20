# foodImageClassfier
Food or non-food? Image Classification with Artificial Neural Network

### Download project
```
git clone https://github.com/enyangxxx/foodImageClassifier.git
```

### Download dataset
1. You can download the dataset Food-5K here by using e.g. Cyberduck to access via FTP:
https://mmspg.epfl.ch/downloads/food-image-datasets/

2. Create a folder 'images' in project root folder:
```
cd foodImageClassfier && mkdir images
```

3. Copy the sub-folders 'training', 'evaluation', 'validation' into the 'images' folder


### Current result
I chose the following hyperparameters:

Number of iterations = 3000
Learning rate = 0.1
Number of layers = 7
Side length of an image = 100
Number of units = side_length*side_length*3, 100, 80, 60, 40, 20, 10, 1

The cost reduction as graph:

<img src="https://github.com/enyangxxx/foodImageClassifier/blob/master/gitImg/cost%20graph.jpg" width="650" height="500">

The cost after each 100th iteration:

<img src="https://github.com/enyangxxx/foodImageClassifier/blob/master/gitImg/costs.jpg" width="350" height="550">

After the training, the accuracy of training, (cross-)validation and test dataset achieved these following values:

#### Training accuracy
![Training accuracy](https://github.com/enyangxxx/foodImageClassifier/blob/master/gitImg/training%20accuracy.jpg)
<img src="https://github.com/enyangxxx/foodImageClassifier/blob/master/gitImg/training%20accuracy.jpg" width="450" height="150">

#### Cross-validation accuracy
![Cross-validation accuracy](https://github.com/enyangxxx/foodImageClassifier/blob/master/gitImg/cv%20accuracy.jpg)

#### Test accuracy
![Test accuracy](https://github.com/enyangxxx/foodImageClassifier/blob/master/gitImg/test%20accuracy.jpg)
