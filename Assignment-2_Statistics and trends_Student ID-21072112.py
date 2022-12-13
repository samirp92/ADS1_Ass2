import numpy as np  # Import the numpy library as np
import pandas as pd  # Import the pandas library as pd
import matplotlib.pyplot as plt  # Import matplotlib.pyplot library as plt
import seaborn as sns  # Import the seaborn library as sns

def read_data(file_name): # Defind a fuction to read the file
    df = pd.read_excel(file_name, header = [3]) # To read the file
    df.drop(['Country Code', 'Indicator Code'], inplace = True, axis = 1) # To remove unnecessary columns
    return df

data = read_data('API_19_DS2_en_excel_v2_4700532.xls') # To store the data
print(data) # To print the data
print(data.describe()) # To show the overall statistics of the data

# Set the data frame (data1) to show trend of Methane emissions(%) for the selected countries
data1 = data[data['Indicator Name'].isin(['Methane emissions (% change from 1990)'])].reset_index(drop = True) # To create data frame with required indicator 
data1 = data1[data1['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                    'Japan','Italy','India','Germany'])].reset_index(drop = True) # To filter the data frame with required counties
data1 = data1.T.reset_index(drop = False)  # To transpose the data
data1 = data1.rename(columns = data1.iloc[0])  # To set the columns name
data1 = data1.iloc[2:].reset_index(drop = True)  # To remove the first two rows
data1.rename({'Country Name' : 'Year'}, axis = 1, inplace = True)  # To remane the first column of data frame
data1['Year'] = pd.to_numeric(data1['Year'])  # To convert the data in numeric
data1 = data1[data1['Year'] >= 2005] # To filter the data data
data1 = data1.dropna().reset_index(drop = True)  # To remove NaN values from data frame
print(data1)  # To print the data frame-1
data1.to_excel('data1.xlsx')  # To save the data frame-1
   
# Set the data frame (data2) to show trend of Nitrous oxide emissions (%) for the selected countries
data2 = data[data['Indicator Name'].isin(['Nitrous oxide emissions (% change from 1990)'])].reset_index(drop = True)  # To create data frame with required indicator 
data2 = data2[data2['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                    'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
data2 = data2.T.reset_index(drop = False)  # To transpose the data
data2 = data2.rename(columns = data2.iloc[0])  # To set the columns name
data2 = data2.iloc[2:].reset_index(drop = True)  # To remove the first two rows
data2.rename({'Country Name' : 'Year'}, axis = 1, inplace = True)  # To remane the first column of data frame
data2['Year'] = pd.to_numeric(data2['Year'])  # To convert the data in numeric
data2 = data2[data2['Year'] >= 2005]  # To filter the data data
data2 = data2.dropna().reset_index(drop = True)  # To remove NaN values from data frame
print(data2)  # To print the data frame-2
data2.to_excel('data2.xlsx')  # To save the data frame-2

# Set the data frame (data3) to show trend of CO2 emissions for the selected countries
data3 = data[data['Indicator Name'].isin(['CO2 emissions (metric tons per capita)'])].reset_index(drop = True)  # To create data frame with required indicator 
data3 = data3[data3['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                     'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
data3 = data3.T.reset_index(drop = False)  # To transpose the data
data3 = data3.rename(columns=data3.iloc[0])  # To set the columns name
data3 = data3.iloc[2:].reset_index(drop = True)  # To remove the first two rows
data3.rename({'Country Name' : 'Year'}, axis=1, inplace = True)  # To remane the first column of data frame
data3['Year'] = pd.to_numeric(data3['Year'])  # To convert the data in numeric
data3 = data3.dropna().reset_index(drop = True)  # To remove NaN values from data frame
print(data3)  # To print the data frame-3
data3.to_excel('data3.xlsx')  # To save the data frame-3

# Set the data frame (data4) to show trend of Urban population for the selected countries
data4 = data[data['Indicator Name'].isin(['Urban population'])].reset_index(drop = True)  # To create data frame with required indicator 
data4 = data4[data4['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                    'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
data4 = data4.T.reset_index(drop = False)  # To transpose the data
data4 = data4.rename(columns = data4.iloc[0])  # To set the columns name
data4 = data4.iloc[2:].reset_index(drop = True)  # To remove the first two rows
data4.rename({'Country Name' : 'Year'}, axis = 1, inplace = True)  # To remane the first column of data frame
data4['Year'] = pd.to_numeric(data4['Year'])  # To convert the data in numeric
data4 = data4.dropna().reset_index(drop = True)  # To remove NaN values from data frame
print(data4)  # To print the data frame-4
data4.to_excel('data4.xlsx')  # To save the data frame-4

def barplot (DataFrame, types, values, xlabel, ylabel, title, name): # Define a function for bar graphs
    DataFrame.plot(kind = types, x = values, rot = '45', width = 0.8, figsize = [18,8], edgecolor = 'black', grid = 1)       
    plt.xlabel(xlabel, fontsize = 18)  # To label the x-axis
    plt.ylabel(ylabel, fontsize = 18)  # To label the y-axis
    plt.title(title, size = 20, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor = (1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.savefig(name, dpi = 300)  # To save the graph
    plt.show()  # To show the graph   

def lineplot (DataFrame, types, values, xlabel, ylabel, title, name): # Define a function for line graphs
    DataFrame.plot(kind = types, x = values, rot = '45', figsize = [18,8], grid = 1)       
    plt.xlim(1990, 2020)   # To set the limit on x-axis
    plt.xlabel(xlabel, fontsize = 18)  # To label the x-axis
    plt.ylabel(ylabel, fontsize = 18)  # To label the y-axis
    plt.title(title, size = 20, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor = (1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.savefig(name, dpi = 300)  # To save the graph
    plt.show()  # To show the graph

# Call the function with required argrumenets to plot bar graphs
barplot(data1, 'bar', 'Year', 'Year', 'Methane emissions (% change)', 'Methane emissions (% change)', 'bar_graph-1') 
barplot(data2, 'bar', 'Year', 'Year', 'Nitrous oxide emissions (% change)', 'Nitrous oxide emissions (% change)', 'bar_graph-2') 
lineplot(data3, 'line', 'Year', 'Year', 'CO2 emissions (metric tons per capita)', 'CO2 emissions (metric tons per capita)', 'line_graph-1') 
lineplot(data4, 'line', 'Year', 'Year', 'Urban population', 'Urban population', 'line_graph-2')

def heat_map():  # Define a function for heat map to find the co-relation with other indicators for perticular county
    data5 = data[data['Indicator Name'].isin(['Population, total', 'Methane emissions (% change from 1990)',
                                        'Nitrous oxide emissions (% change from 1990)', 'Energy use (kg of oil equivalent per capita)',                               
                                        'PFC gas emissions (thousand metric tons of CO2 equivalent)',
                                        'Agricultural land (sq. km)'])].reset_index(drop = True)  # To create data frame with required indicator  
    data5 = data5[data5['Country Name'].isin(['India'])].reset_index(drop = True)  # To filter the data frame with required country
    data5 = data5.T.reset_index(drop = False)  # To transpose the data
    data5 = data5.rename(columns = data5.iloc[1])  # To set the columns name
    data5 = data5.iloc[2:].reset_index(drop = True)  # To remove the two rows
    data5.rename(columns = {'Energy use (kg of oil equivalent per capita)' : 'Energy use', 
                      'PFC gas emissions (thousand metric tons of CO2 equivalent)' : 'PFC gas emissions',
                      'Nitrous oxide emissions (% change from 1990)' : 'Nitrous oxide emissions (%)',
                      'Methane emissions (% change from 1990)' : 'Methane emissions (%)'},inplace = True)  # To remname the column name to fit into heat map image
    data5 = data5.astype(float)  # To convert data into float value
    data5 = data5.iloc[:,1:]  # To remove first column
    data5 = data5.dropna().reset_index(drop = True)  # To remove NaN valuses form data frame
    print(data5)  # To print the data frame-5
    data5.to_excel('data5.xlsx')  # To save the data frame-5
    plt.figure(figsize = [42,22])  # To set the figure size
    heat_map1 = sns.heatmap(data5.corr(), vmin = -1, vmax = 1, annot = True, linewidths = 0.5, linecolor = "black", annot_kws = {"size": 30})  # To plot heat map
    heat_map1.set_xticklabels(heat_map1.get_xticklabels(), rotation = 25, 
                              horizontalalignment = 'center', fontweight = 'light', fontsize = '30')  # To label the x-axis
    heat_map1.set_yticklabels(heat_map1.get_yticklabels(), rotation = 25, 
                              horizontalalignment = 'right', fontweight = 'light', fontsize = '30')   # To label the y-axis
    plt.title('India_heat map', size = 40, color = 'black')  # To give title of graph
    plt.savefig('heat_map.jpeg', dpi = 300)  # To save the heat map
    plt.show()  # To show the graph
    
def pie_chart():  # Define a function for pie chart to find the corelation between Electric power consumption and Urban Population
    data7 = data[data['Indicator Name'].isin(['Electric power consumption (kWh per capita)'])].reset_index(drop = True)  # To create data frame  
    data7 = data7[data7['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    data7['Total Electric power consumption'] = data7.iloc[:,2:].sum(axis = 1)  # To make new column with sum of all data
    print(data7)  # To print the data frame-7
    data7.to_excel('data7.xlsx')  # To save the data frame-7
    data8 = data[data['Indicator Name'].isin(['Urban population'])].reset_index(drop = True)  # To create data frame with required indicator 
    data8 = data8[data8['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    data8['Total Urban population'] = data8.iloc[:,2:].sum(axis = 1)  # To make new column with sum of all data
    print(data8)  # To print the data frame-8
    data8.to_excel('data8.xlsx')  # To save the data frame-8
    plt.figure(figsize = ([18, 9]))  # To set the figure resolution
    plt.suptitle('Electric power consumption vs Total urban population', size = 25, color = 'black')  # To add title of plot
    plt.subplot(1,2,1)  # To set up the subplot
    plt.pie(data7['Total Electric power consumption'], labels = data7['Country Name'], autopct = '%1.0f%%', startangle = 90, 
            rotatelabels = False, counterclock = False, pctdistance = 0.75, labeldistance  = 1.05, radius = 1, textprops = {'size' : 20})  # To plot pie chart-1
    plt.title('Total Electric power consumption', size = 22, color = 'black')  # To set the title of pie chart-1
    plt.subplot(1,2,2)  # To set up the subplot
    plt.pie(data8['Total Urban population'], labels = data8['Country Name'], autopct = '%1.0f%%', startangle = 90, 
            rotatelabels = False, counterclock = False, pctdistance = 0.75, labeldistance  = 1.05, radius = 1, textprops = {'size' : 20})  # To plot pie chart-2
    plt.title('Total Urban population', size = 22, color = 'black')  # To set the title of pie chart-2
    plt.savefig('pie_chart.jpeg', dpi = 300)  # To save the pie chart
    plt.show()  # To show the graph

def data_table():  # Define a function to generate the data in table form for the Total greenhouse gas emission wrt diffirent countries
    data9 = data[data['Indicator Name'].isin(['Total greenhouse gas emissions (kt of CO2 equivalent)'])].reset_index(drop = True)  # To create data frame 
    data9 = data9[data9['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    data9 = data9.iloc[:,[0,32,37,42,47,52,57]]  # To filter the specific columns for five con
    data9['Total greenhouse gas emissions (%)'] = data9.iloc[:,1:].sum(axis=1)  # To sum the all values and make new column
    data9['Total greenhouse gas emissions (%)'] = data9['Total greenhouse gas emissions (%)']/data9['Total greenhouse gas emissions (%)'].sum() * 100  # To create %
    data9 = data9.round(decimals = 0)  # To round up the value after decimal
    print(data9)  # To print the data frame-9
    data9.to_excel('data9.xlsx')  # To save the data frame-9

data_table()
pie_chart()
heat_map()
