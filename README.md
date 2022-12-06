# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.
In this project, I have created and optimize an ML pipeline. I used standard Scikit-learn Logistic Regression—the hyperparameters of which I have optimized using HyperDrive. Its then fine tuned using hyper parameter and then using AutoML submitted and compared results of which model ie best fit

You can see the main steps  in the diagram below:
![image](https://github.optum.com/storage/user/2531/files/209cfa90-aefc-40e8-8c36-60136beef67f)
Source:- [img src](https://learn.udacity.com/nanodegrees)

These are the steps for the project execution 
![image](https://github.optum.com/storage/user/2531/files/4b064f02-8e77-4b9d-a215-b2c740e49912)
Source: [img src](Https://learn.udacity.com/nanodegrees)

- Create tabular data set.
- Split data into train and test set(in my case have taken 70:30).
- Specify Parameter Sampler.
- Specify Policy.
- Using hyperdrive config.
- Using AutoML configuration set AutomML config.
- Submit experiment to find bet model and best fit.



## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
Goal of this project is to predict wheather client will subscribe for a bank term deposit. Data set has different features like age employment and marital status.

Scikit-learn pipeline gave an accuracy of **0.9162377 HyperDrive Model

![image](https://github.optum.com/storage/user/2531/files/b6d5a871-4a53-4053-b21c-aacf99a00b34)

**AutoML model**  has accuracy of **0.91657
![image](https://github.optum.com/storage/user/2531/files/973cef58-2021-4e8c-ab9b-bba1cdf6678c)



## Scikit-learn Pipeline

  ## Parameter sampler

Selection was random Sampling specified the parameter sampler as such:

      ps = RandomParameterSampling(
          {
              '--C' : choice(0.001,0.01,0.1,1,10,20,50,100,200,500,1000),
              '--max_iter': choice(50,100,200,300)
          }
      )

Random paramaeter sample is quicker and close to accurate, while gridParameterSamplingis most accurate but very exhastive, if Compute is not a concern this method can be used.
 

## Stopping Policy
Terminating poorly performing model is extreemly important when considering to time and compute aspect. If the error rate is not with in normal range then it should be terminated. In this project I have used 

```python
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```
*slack facor specifies ratio to which model should allow before terminating.* 
 
 
## AutoML
AutoML is the process where ML is automated to run all different model and identify which can be best fit for the given data set.

Below is the configration where we speicfy for automl to set and run.

```python
automl_config = AutoMLConfig(
    task='classification',
    iterations=30,
    label_column_name='y',
    iteration_timeout_minutes=5,
    primary_metric='accuracy',
    training_data=tds,
    n_cross_validations=2)
```

## Pipeline comparison
Two Model we are going to compare are **HyperDrive and AutoML Model** 

| Hyperdrive Model        |            |
| ----------------------- |------------|
|  ID                     | HD_bee75b8a-833d-4dfe-b055-dc4cc6e38939_8 |
| Parameter|--C 0.1 --max_iter 300|
| Accuracy      |  0.9162367     |


| AutoML Model        |            |
| ----------------------- |------------|
|  ID                     | AutoML_73f743ce-78a2-4706-a41e-17d1fa5b3d1e |
| Parameter| task='classification',iterations=30,label_column_name='y',iteration_timeout_minutes=5,primary_metric='accuracy',training_data=tds,  n_cross_validations=2|
|Algortithm |VotingEnsemble|
| Accuracy      |  0.91657    |

AutoML will always be best suited for the reason it will run all different avilable algortithms rather than some one deciding and running each model and then adjust parameter to give better model


## Future work
The most common issue with any model is with reagarding to check for Bias. Yes model can be only as good as data set can be. Careful evaluation of dataset needs to be done to see model is performing well enough. 


## References

- [https://github.com/Azure/MachineLearningNotebooks/issues/1263]
- [https://towardsdatascience.com/hidden-tricks-for-running-automl-experiment-from-azure-machine-learning-sdk-915d4e3f840e]

