import numpy as np  # Import the numpy library as np
import pandas as pd  # Import the pandas library as pd
import matplotlib.pyplot as plt  # Import matplotlib.pyplot library as plt
import seaborn as sns  # Import the seaborn library as sns


df = pd.read_excel('API_19_DS2_en_excel_v2_4700532.xls', header = [3])   # To load excel file
df.drop(['Country Code', 'Indicator Code'], inplace = True, axis = 1)  # To drop the unnecessary columns


def bar_graph1():  # Define a function for bar graph to show trend of Methane emissions(%) for the selected countries
    df1 = df[df['Indicator Name'].isin(['Methane emissions (% change from 1990)'])].reset_index(drop = True) # To create data frame with required indicator 
    df1 = df1[df1['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True) # To filter the data frame with required counties
    df1 = df1.T.reset_index(drop = False)  # To transpose the data
    df1 = df1.rename(columns = df1.iloc[0])  # To set the columns name
    df1 = df1.iloc[2:].reset_index(drop = True)  # To remove the first two rows
    df1.rename({'Country Name' : 'Year'}, axis = 1, inplace = True)  # To remane the first column of data frame
    df1['Year'] = pd.to_numeric(df1['Year'])  # To convert the data in numeric
    df1 = df1[df1['Year'] >= 2005] # To filter the data data
    df1 = df1.dropna().reset_index(drop = True)  # To remove NaN values from data frame
    print(df1)  # To print the data frame-1
    df1.to_excel('df1.xlsx')  # To save the data frame-1
    df1.plot.bar(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], 
                 rot = '45', width = 0.8, figsize = [18,8], edgecolor = "black", grid = 1)  # To plot bar chart        
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('Methane emissions (% change)', fontsize = 15)  # To label the y-axis
    plt.title('Methane emissions (% change)', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor = (1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.savefig('bar_graph-1.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph


def bar_graph2():  # Define a function for bar graph to show trend of Nitrous oxide emissions(%) for the selected countries
    df2 = df[df['Indicator Name'].isin(['Nitrous oxide emissions (% change from 1990)'])].reset_index(drop = True)  # To create data frame with required indicator 
    df2 = df2[df2['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    df2 = df2.T.reset_index(drop = False)  # To transpose the data
    df2 = df2.rename(columns = df2.iloc[0])  # To set the columns name
    df2 = df2.iloc[2:].reset_index(drop = True)  # To remove the first two rows
    df2.rename({'Country Name' : 'Year'}, axis = 1, inplace = True)  # To remane the first column of data frame
    df2['Year'] = pd.to_numeric(df2['Year'])  # To convert the data in numeric
    df2 = df2[df2['Year'] >= 2005]  # To filter the data data
    df2 = df2.dropna().reset_index(drop = True)  # To remove NaN values from data frame
    print(df2)  # To print the data frame-2
    df2.to_excel('df2.xlsx')  # To save the data frame-2
    df2.plot.bar(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], 
                 rot = '45', width = 0.8, figsize = [18,8], edgecolor = "black", grid = 1)  # To plot bar chart      
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('Nitrous oxide emissions (% change)', fontsize = 15)  # To label the y-axis
    plt.title('Nitrous oxide emissions (% change)', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor = (1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.savefig('bar_graph-2.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph


def line_graph1():  # Define a function for line graph to show trend of CO2 emissions for the selected countries
    df3 = df[df['Indicator Name'].isin(['CO2 emissions (metric tons per capita)'])].reset_index(drop = True)  # To create data frame with required indicator 
    df3 = df3[df3['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    df3 = df3.T.reset_index(drop = False)  # To transpose the data
    df3 = df3.rename(columns=df3.iloc[0])  # To set the columns name
    df3 = df3.iloc[2:].reset_index(drop = True)  # To remove the first two rows
    df3.rename({'Country Name' : 'Year'}, axis=1, inplace = True)  # To remane the first column of data frame
    df3['Year'] = pd.to_numeric(df3['Year'])  # To convert the data in numeric
    df3 = df3.dropna().reset_index(drop = True)  # To remove NaN values from data frame
    print(df3)  # To print the data frame-3
    df3.to_excel('df3.xlsx')  # To save the data frame-3
    df3.plot(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], figsize = [18,8], grid = 1)  # To plot line graph       
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('CO2 emissions (metric tons per capita)', fontsize = 15)  # To label the y-axis
    plt.title('CO2 emissions (metric tons per capita)', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor = (1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.savefig('line_graph-1.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph


def line_graph2():  # Define a function for line graph to show trend of Urban population for the selected countries
    df4 = df[df['Indicator Name'].isin(['Urban population'])].reset_index(drop = True)  # To create data frame with required indicator 
    df4 = df4[df4['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    df4 = df4.T.reset_index(drop = False)  # To transpose the data
    df4 = df4.rename(columns = df4.iloc[0])  # To set the columns name
    df4 = df4.iloc[2:].reset_index(drop = True)  # To remove the first two rows
    df4.rename({'Country Name' : 'Year'}, axis = 1, inplace = True)  # To remane the first column of data frame
    df4['Year'] = pd.to_numeric(df4['Year'])  # To convert the data in numeric
    df4 = df4.dropna().reset_index(drop = True)  # To remove NaN values from data frame
    print(df4)  # To print the data frame-4
    df4.to_excel('df4.xlsx')  # To save the data frame-4
    df4.plot(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], figsize = [18,8], grid = 1)  # To plot line graph
    plt.xlim(1990, 2020)   # To set the limit on x-axis        
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('Urban population', fontsize = 15)  # To label the y-axis
    plt.title('Urban population', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor = (1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15) # To increase the xtick font size
    plt.rc('ytick', labelsize = 15) # To increase the ytick font size
    plt.savefig('line_graph-2.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph


def heat_map():  # Define a function for heat map to find the co-relation with other indicators for perticular conutry
    df5 = df[df['Indicator Name'].isin(['Population, total', 'Methane emissions (% change from 1990)',
                                        'Nitrous oxide emissions (% change from 1990)', 'Energy use (kg of oil equivalent per capita)',                               
                                        'PFC gas emissions (thousand metric tons of CO2 equivalent)',
                                        'Agricultural land (sq. km)'])].reset_index(drop = True)  # To create data frame with required indicator  
    df5 = df5[df5['Country Name'].isin(['India'])].reset_index(drop = True)  # To filter the data frame with required country
    df5 = df5.T.reset_index(drop = False)  # To transpose the data
    df5 = df5.rename(columns = df5.iloc[1])  # To set the columns name
    df5 = df5.iloc[2:].reset_index(drop = True)  # To remove the two rows
    df5.rename(columns = {'Energy use (kg of oil equivalent per capita)' : 'Energy use', 
                      'PFC gas emissions (thousand metric tons of CO2 equivalent)' : 'PFC gas emissions',
                      'Nitrous oxide emissions (% change from 1990)' : 'Nitrous oxide emissions (%)',
                      'Methane emissions (% change from 1990)' : 'Methane emissions (%)'},inplace = True)  # To remname the column name to fit into heat map image
    df5 = df5.astype(float)  # To convert data into float value
    df5 = df5.iloc[:,1:]  # To remove first column
    df5 = df5.dropna().reset_index(drop = True)  # To remove NaN valuses form data frame
    print(df5)  # To print the data frame-5
    df5.to_excel('df5.xlsx')  # To save the data frame-5
    plt.figure(figsize = [42,22])  # To set the figure size
    heat_map1 = sns.heatmap(df5.corr(), vmin = -1, vmax = 1, annot = True, linewidths = 0.5, linecolor = "black", annot_kws = {"size": 30})  # To plot heat map
    heat_map1.set_xticklabels(heat_map1.get_xticklabels(), rotation = 25, 
                              horizontalalignment = 'center', fontweight = 'light', fontsize = '30')  # To label the x-axis
    heat_map1.set_yticklabels(heat_map1.get_yticklabels(), rotation = 25, 
                              horizontalalignment = 'right', fontweight = 'light', fontsize = '30')   # To label the y-axis
    plt.title('India_heat map', size = 40, color = 'black')  # To give title of graph
    plt.savefig('heat_map.jpeg', dpi = 300)  # To save the heat map
    plt.show()  # To show the graph
    

def pie_chart():  # Define a function for pie chart to find the corelation between Electric power consumption and Urban Population
    df7 = df[df['Indicator Name'].isin(['Electric power consumption (kWh per capita)'])].reset_index(drop = True)  # To create data frame with required indicator 
    df7 = df7[df7['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    df7['Total Electric power consumption'] = df7.iloc[:,2:].sum(axis = 1)  # To make new column with sum of all data
    print(df7)  # To print the data frame-7
    df7.to_excel('df7.xlsx')  # To save the data frame-7
    df8 = df[df['Indicator Name'].isin(['Urban population'])].reset_index(drop = True)  # To create data frame with required indicator 
    df8 = df8[df8['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    df8['Total Urban population'] = df8.iloc[:,2:].sum(axis = 1)  # To make new column with sum of all data
    print(df8)  # To print the data frame-8
    df8.to_excel('df8.xlsx')  # To save the data frame-8
    plt.figure(figsize = ([18, 9]))  # To set the figure resolution
    plt.suptitle('Electric power consumption vs Total urban population', size = 25, color = 'black')  # To add title of plot
    plt.subplot(1,2,1)  # To set up the subplot
    plt.pie(df7['Total Electric power consumption'], labels = df7['Country Name'], autopct = '%1.0f%%', startangle = 90, 
            rotatelabels = False, counterclock = False, pctdistance = 0.75, labeldistance  = 1.05, radius = 1, textprops = {'size' : 20})  # To plot pie chart-1
    plt.title('Total Electric power consumption', size = 22, color = 'black')  # To set the title of pie chart-1
    plt.subplot(1,2,2)  # To set up the subplot
    plt.pie(df8['Total Urban population'], labels = df8['Country Name'], autopct = '%1.0f%%', startangle = 90, 
            rotatelabels = False, counterclock = False, pctdistance = 0.75, labeldistance  = 1.05, radius = 1, textprops = {'size' : 20})  # To plot pie chart-2
    plt.title('Total Urban population', size = 22, color = 'black')  # To set the title of pie chart-2
    plt.savefig('pie_chart.jpeg', dpi = 300)  # To save the pie chart
    plt.show()  # To show the graph


def data_table():  # Define a function to generate the data in table form for the Total greenhouse gas emission wrt diffirent countries
    df9 = df[df['Indicator Name'].isin(['Total greenhouse gas emissions (kt of CO2 equivalent)'])].reset_index(drop = True)  # To create data frame with required indicator 
    df9 = df9[df9['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop = True)  # To filter the data frame with required counties
    df9 = df9.iloc[:,[0,32,37,42,47,52,57]]  # To filter the specific columns for five con
    df9['Total greenhouse gas emissions (%)'] = df9.iloc[:,1:].sum(axis=1)  # To sum the all values and make new column
    df9['Total greenhouse gas emissions (%)'] = df9['Total greenhouse gas emissions (%)']/df9['Total greenhouse gas emissions (%)'].sum() * 100  # To convert the data wrt percentage
    df9 = df9.round(decimals = 0)  # To round up the value after decimal
    print(df9)  # To print the data frame-9
    df9.to_excel('df9.xlsx')  # To save the data frame-9


bar_graph1()
bar_graph2()
data_table()
pie_chart()
line_graph1()
line_graph2()
heat_map()
