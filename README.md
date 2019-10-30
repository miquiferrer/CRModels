# CRModels

This repository cointains the necessary files to build 3 different models for the completion rate
prediction model and serve one of them via a REST API using Flask. The three models are implemened in 
Python and sotred in a proper project structure. In addition to that, there is a Jupyter notebook with an initial 
data exploration and model training with some plots. All the code of the jupyter notebook is also spread in the Python implementation.

## Models

As mentioned above, there are 3 different models with the purpose of tackling the problem from different perspectives. 
In all cases the problem has been treated as a regression problem, where the predicted value is continuous (i.e. the 
completion rate)

* **Model 1:** Simple linea regression model using Scikit-learn
* **Model 2:** Random forest based on XGBoost
* **Model 3:** "Deep" Neural Network implemented in Tensorflow

**Note:** The third model is the one being served via a REST API.

The rationale behind using these 3 models is the following: With the first model, we wanted to show a very simple approach to linear regression. With the second model, we resort to a more complex model (gradient boosting). Here we see cross validation which we did not see before. Finally, with the third model, we use a Deep Neural Network and serve this model via a REST API.


## Dataset

Due to Github limitations, the maximum file size allowed to be uploaded is 100Mb. Since the original dataset id bigger 
than that (> 200Mb) and we wanted for this repo to be self contained, the dataset used to build the training and test 
datasets is a fraction (roughly 20%) of the original dataset. That is, the original dataset has more than 1M rows. From 
this dataset we've chosen 200k rows randonly chosen. Fromt this 200k-row dataset, we build both the training ans test set 
using 100k rows randonly chosen as well. Out of these 100k rows, 80k rows correspond to the training set and the remaining 20k rows are assigned to the test set. That is, the test set is 20% of the 100k dataset and the remaining 80% is assigned to the training set.


## Execution

In the following we explain the necessary steps in order to, from the 100k dataset, build the training and the test sets, train and evaluate the 3 models, and start the API and query the third model.
 
### Prepare environment

* Clone the Git repository
````bash
$ git clone git@github.com:miquiferrer/CRModels.git
````
* (Optional) If possible, create a new virtual environment in order not to pollute the OS Python installation
```bash
$ mkvirtualenv TFTest
```

* Install the Python requirements
```bash
$ pip install -r requirements.txt
```

### Prepare data

* Once in the project folder, get into the "src" folder
```bash
$ cd src
```
* Preprocessing original data. Input: 100k dataset. Output: Clean 100k dataset.
````bash
$ python -m data_processing.load_and_clean_data
````

* Generate the training and test sets
````bash
$ python -m data_processing.generate_data_sets
````

### Build and train the models

* Build, train and evaluate the first model (i.e. Linear Regression with Scikit learn)
```bash
$ python -m models.sklearn_linear_regression
```

* Build, train and evaluate the second model (i.e. XGBoost regressor)
````bash
$ python -m models.xgboost_regression
````

* Build, train and evaluate the third model (DNN using Tensorflow)
```bash
$ python -m models.tf_dnn_regression
```

This will also perform a prediction using the following input:

[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0]

In order to test the exported model, we will perform the same prediction but using the exported model (this will serve us to verify that the exported model is the same DNN trained model). We can query the model with the following command:

````bash
saved_model_cli run --dir ../data/TFModel/1/ --tag_set serve --signature_def serving_default --input_exprs='dense_input=np.array([[0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1., 2.]])'
```` 

**Note:** Notice that the three models have similar metric results and looking at the Jupyter notebook, we see that the Deep Neural  model is overfitting as the metric for the validation set gets deteriorated as epochs advance whereas the metrics improve for the training set.

### Serve third model via REST api

* Start Flask
````bash
$ python -m api.app
````
This will load the stored model and start the web servce at http://0.0.0.0:5000. Thus, if you enter "http://0.0.0.0:5000" in your web browser, you should see the following message "Congratulations! Flask is properly running!"

* Query the model
````bash
curl -d '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0]' -H "Content-Type: application/json" -X POST http://0.0.0.0:5000/api
````
 
As you can see, this gives the same result as before. :-)


