# graphical-user-interface

Runnable instance
After importing the project, run the main.py program, and the GUI interface will pop up,
First, click the "..." button to select the data table, and then click the submit button（There are data tables in the project, which represent the characteristic values of magnolol with different concentrations and can be downloaded directly）
After the 'please enter y' text, enter the symbol of the column representing the concentration, pre
Enter the selected eigenvalue after "please enter X", F + AP + AQ
After you click Submit, you can click the 'plot' button to view the image results, and you can click the 'result' button to view the predicted value of the model.
In the right half area, click "look at the relationship between Y and X" to view the correlation between Y and X
Enter F + AP + AQ after 'let is take x that has a linear ralationship',
The 'plot'  buttons and 'result' buttons are used to view the optimized results
Similarly, the data table named 'data1' represents the data of honokiol, and the analysis process is the same as the data table named 'data'

Function
It is very difficult for non analysts to predict the concentration of Chinese patent medicine by establishing mathematical model based on EEM.A multiple linear regression model is established,
A GUI interface based on Python is developed. For non professionals, the mathematical model can be quickly established through this visual interface to predict the concentration of a certain component in the mixture. The program greatly reduces the difficulty of component analysis of Chinese patent medicine and improves the analysis efficiency.


Course
First import the following version of the library file in Python 3
easygui=0.95.2，matplotlib=3.3.4，seaborn=0.11.1，sns=0.1，statsmodels=0.12.2，xlrd=1.2.0
After importing the project, run the main.py program, and the GUI interface will pop up,First, the interface requires submission of an excel file containing the eigenvalues as well as the concentration values of single substances, which is in the format of table1 in Section 2.3 of 《The application of multilinear regression model for quantitative analysis on the basis of EEM spectra and the release of a free graphical user interface》
Enter the sign of the column representing the concentration in the interface,
Enter the symbol of the column representing the feature value in the interface, split with '+'.
After clicking on the 'submit' button and further clicking on the plot button, the graph result was obtained and the prediction value was obtained by clicking on the 'resule' button.
In the right half area, the correlation between Y and X can be viewed, and variables with strong correlation can be filtered out, with "" + "" making the segmentation, the sameness, and clicking on the plot and result buttons can view the final result.
