 # SystemRequest

 ## DatasetCreation:
 * api: call api using request lib
 * data: data received from vSDS team
 * data_creation: data created for matching (functional) problem, text2table problem
 * data_prepare: prepared data after prompting and before training
 * demo: demo the system before deploying
 * deploy: building docker image for deploying on GPU environment
 * deploy_support: prepare for deploy (load model)
 * process: process raw data from folder "data"
 * prompt: generate data using GPT api

 Order to view and follow: 
 * generate data: data -> process -> prompt -> data_prepare -> data_creation
 * call api: api
 * system view: demo -> deploy, deploy_support

 ## Achatbot
 * data: data for training Text2Table model
 * output: output of training process
 * text2table: python file for training Text2Table model
 * matching: run each .ipynb in drive for training 
 * file Text2TableTraining.ipynb: run all in kaggle for training.


