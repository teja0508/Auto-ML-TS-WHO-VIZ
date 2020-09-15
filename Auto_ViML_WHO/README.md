############# This is not an Officially Supported Google Product! #########################################
# Auto_ViML_WHO
   Automated Variant Interpretable Machine Learning project with Hyper Opt (WHO).
   Build Multiple, Interpretable, ML Models Fast. Now using Hyper Opt.

    #########################################################################################################
    #Copyright 2019 Google LLC                                                                        #######
    #                                                                                                 #######
    #Licensed under the Apache License, Version 2.0 (the "License");                                  #######
    #you may not use this file except in compliance with the License.                                 #######
    #You may obtain a copy of the License at                                                          #######
    #                                                                                                 #######
    #    https://www.apache.org/licenses/LICENSE-2.0                                                  #######
    #                                                                                                 #######
    #Unless required by applicable law or agreed to in writing, software                              #######
    #distributed under the License is distributed on an "AS IS" BASIS,                                #######
    #WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         #######
    #See the License for the specific language governing permissions and                              #######
    #limitations under the License.                                                                   #######
    #########################################################################################################
    ####       Automatically Build Variant Interpretable Machine Learning Models (Auto_ViML)           ######
    ####                                Developed by Ramadurai Seshadri                                ######
    #########################################################################################################
    ####   Auto_ViML was designed for building a High Performance Interpretable Model With Fewest Vars.   ###
    ####   The "V" in Auto_ViML stands for Variant because it tries Multiple Models and Multiple Features ###
    ####   to find the Best Performing Model for any data set.The "i" in Auto_ViML stands " Interpretable"###
    ####   since it selects the fewest Features to build a simpler, more interpretable model. This is key. ##
    ####   Auto_ViML is built mostly using Scikit-Learn, Numpy, Pandas and Matplotlib. Hence it should run ##
    ####   on any Python 2 or Python 3 Anaconda installations. You won't have to import any special      ####
    ####   Libraries other than "SHAP" library for SHAP values which provides more interpretability.    #####
    ####   But if you don't have it, Auto_ViML will skip it and show you the regular feature importances. ###
    #########################################################################################################
    ####   INPUTS:                                                                                        ###
    #########################################################################################################
    ####   train: could be a datapath+filename or a dataframe. It will detect which is which and load it.####
    ####   test: could be a datapath+filename or a dataframe. If you don't have any, just leave it as "". ###  
    ####   submission: must be a datapath+filename. If you don't have any, just leave it as empty string.#### 
    ####   target: name of the target variable in the data set.                                          ####
    ####   sep: if you have a spearator in the file such as "," or "\t" mention it here. Default is ",". ####
    ####   scoring_parameter: if you want your own scoring parameter such as "f1" give it here. If not, #####
    ####       it will assume the appropriate scoring param for the problem and it will build the model.#####
    ####   Boosting Flag: you have 3 possible choices (default is False):                               #####
    ####    None = This will build a Linear Model                                                       #####
    ####    False = This will build a Random Forest or Extra Trees model (also known as Bagging)        #####
    ####    True = This will build an XGBoost model                                                     #####
    ####   Add_Poly: Default is 0. It has 2 additional settings:                                        #####
    ####    1 = Add interaction variables only such as x1*x2, x2*x3,...x9*10 etc.                       ##### 
    ####    2 = Add Interactions and Squared variables such as x1**2, x2**2, etc.                       #####
    ####   Stacking_Flag: Default is False. If set to True, it will add an additional feature which     #####
    ####         is derived from predictions of another model. This is used in some cases but may result#####
    ####         in overfitting. So be careful turning this flag "on".                                  #####
    ####   Binning_Flag: Default is False. It set to True, it will convert the top numeric variables    #####
    ####         into binned variables through a technique known as "Entropy" binning. This is very     #####
    ####         helpful for certain datasets (especially hard to build models).                        #####
    ####   Imbalanced_Flag: Default is False. If set to True, it will downsample the "Majority Class"   #####
    ####         in an imbalanced dataset and make the "Rare" class at least 5% of the data set. This   #####
    ####         the ideal threshold in my mind to make a model learn. Do it for Highly Imbalanced data.#####
    ####   verbose: This has 3 possible states:                                                         #####
    ####    0 = limited output. Great for running this silently and getting fast results.               #####
    ####    1 = more charts. Great for knowing how results were and making changes to flags in input.   #####
    ####    2 = lots of charts and output. Great for reproducing what Auto_ViML does on your own.       #####
    #########################################################################################################
    ####   OUTPUTS:                                                                                     #####
    #########################################################################################################
    ####   model: It will return your trained model                                                     #####
    ####   features: the fewest number of features in your model to make it perform well                #####
    ####   train_modified: this is the modified train dataframe after removing and adding features      #####
    ####   test_modified: this is the modified test dataframe with the same transformations as train    #####
    ####   Finally, it writes your submission file to disk in the current directory called "mysubmission.csv"
    ####   This submission file is ready for you to show it clients or submit it to competitions.       #####     
    ####   If no submission file was given but as long as you give it a test file name, it will create  #####
    ####   a submission file for you named "mySubmission.csv".                                          #####
    #################               A D D I T I O N A L    N O T E S                              ###########
    ####   Auto_ViML works on any Multi-Class, Multi-Label Data Set. So you can have many labels.       #####  
    ####   Scoring Metric Suggestions:                                                                  #####
    ####   If you have Binary Class and Multi-Class in a Single Label, Choose Accuracy. It will        ######
    ####   do very well. If you want something better, try roc_auc even for Multi-Class which works.   ######
    ####   You can try F1 or Weighted F1 if you want something complex or for Multi-Class.             ######
    ####   Note that For Imbalanced Classes (<=5% classes), it automatically adds Class Weights.       ######
    ####   Also, Note that it handles Multi-Label automatically so you can send Train data             ######
    ####   with multiple Labels (Targets) and it will automatically predict for each Label.            ######
    ####   Finally this is Meant to Be a Fast Algorithm, so use it for just quick POCs                 ######
    ####   This is Not Meant for Production Problems. It produces great models but may not be Perfect! ######
    #########################       G  O  O  D    L   U   C   K!                   ##########################
    #########################################################################################################

# Steps:
1. Copy or download this Auto_ViML.py file to any directory. 
1. In the same directory, open a Jupyter Notebook and use this line to import the .py file:

    <code>from Auto_ViML import Auto_ViML</code>

1. Then use Pandas to import a CSV or other data file into a dataframe and split it into Train and Test data frames. Notice that Auto_ViML uses DataFrames as inputs. Do not send in Numpy Arrays since it will give an error.

1. Finally, call Auto_ViML using the train, test dataframes and the name of the target variable in data frame. That's all.

    <code>Auto_ViML(train, target, test='',sample_submission='',modeltype='Classification',
            scoring_parameter='logloss', Boosting_Flag=None,
            Add_Poly=0, Stacking_Flag=False, Binning_Flag=False,
              Imbalanced_Flag=False, verbose=0)</code>
              
Hope this helps. You don't need any special Libraries other than whatever is in your Anaconda Python Distribution. 

Also, BTW, Auto_ViML runs on Python 3 versions without problems. But I suspect it can run on Python 2 as well.

<b>License</b>
Apache 2.0; see <a href="https://github.com/mbrukman/autogen/blob/master/LICENSE.txt">LICENSE.txt</a> for details.

<b>Disclaimer</b>
This project is not an official Google project. It is not supported by Google and Google specifically disclaims all warranties as to its quality, merchantability, or fitness for a particular purpose.
