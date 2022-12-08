import numpy as np  # Import the numpy library as np
import pandas as pd  # Import the pandas library as pd
import matplotlib.pyplot as plt  # Import matplotlib.pyplot library as plt


df = pd.read_excel('API_19_DS2_en_excel_v2_4700532.xls', header=[3])   # To load excel file
df.drop(['Country Code', 'Indicator Code'], inplace = True, axis = 1)  # To drop the unnecessary columns


def bar_graph1():  # Define a function for bar graph to show trend of Methane emissions for the selected countries
    df1 = df[df['Indicator Name'].isin(['Methane emissions (% change from 1990)'])].reset_index(drop=True) # To create data frame with required indicator 
    df1 = df1[df1['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop=True) # To filter the data frame with required counties
    df1 = df1.T.reset_index(drop=False)  # To transpose the data
    df1 = df1.rename(columns=df1.iloc[0])  # To set the columns name
    df1 = df1.iloc[2:].reset_index(drop=True)  # To remove the first two rows
    df1.rename({'Country Name': 'Year'}, axis=1, inplace=True)  # To remane the first column of data frame
    df1['Year'] = pd.to_numeric(df1['Year'])  # To convert the data in numeric
    df1 = df1[df1['Year'] >= 2005] # To filter the data data
    df1 = df1.dropna().reset_index(drop=True)  # To remove NaN values from data frame
    print(df1)  # To print the data frame-1
    df1.to_excel('df1.xlsx')  # To save the data frame-1
    df1.plot.bar(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], 
                 rot = '45', width = 0.8, figsize = [18,8], edgecolor = "black")  # To plot bar chart        
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('Methane emissions (% change)', fontsize = 15)  # To label the y-axis
    plt.title('Methane emissions (% change)', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.savefig('bar_graph-1.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph


def bar_graph2():  # Define a function for bar graph to show trend of Nitrous oxide emissions for the selected countries
    df2 = df[df['Indicator Name'].isin(['Nitrous oxide emissions (% change from 1990)'])].reset_index(drop=True)  # To create data frame with required indicator 
    df2 = df2[df2['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop=True)  # To filter the data frame with required counties
    df2 = df2.T.reset_index(drop=False)  # To transpose the data
    df2 = df2.rename(columns=df2.iloc[0])  # To set the columns name
    df2 = df2.iloc[2:].reset_index(drop=True)  # To remove the first two rows
    df2.rename({'Country Name': 'Year'}, axis=1, inplace=True)  # To remane the first column of data frame
    df2['Year'] = pd.to_numeric(df2['Year'])  # To convert the data in numeric
    df2 = df2[df2['Year'] >= 2005]  # To filter the data data
    df2 = df2.dropna().reset_index(drop=True)  # To remove NaN values from data frame
    print(df2)  # To print the data frame-2
    df2.to_excel('df2.xlsx')  # To save the data frame-2
    df2.plot.bar(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], 
                 rot = '45', width = 0.8, figsize = [18,8], edgecolor = "black")  # To plot bar chart      
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('Nitrous oxide emissions (% change)', fontsize = 15)  # To label the y-axis
    plt.title('Nitrous oxide emissions (% change)', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.savefig('bar_graph-2.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph

    
def line_graph1():  # Define a function for line graph to show trend of CO2 emissions for the selected countries
    df3 = df[df['Indicator Name'].isin(['CO2 emissions (metric tons per capita)'])].reset_index(drop=True)  # To create data frame with required indicator 
    df3 = df3[df3['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop=True)  # To filter the data frame with required counties
    df3 = df3.T.reset_index(drop=False)  # To transpose the data
    df3 = df3.rename(columns=df3.iloc[0])  # To set the columns name
    df3 = df3.iloc[2:].reset_index(drop=True)  # To remove the first two rows
    df3.rename({'Country Name': 'Year'}, axis=1, inplace=True)  # To remane the first column of data frame
    df3['Year'] = pd.to_numeric(df3['Year'])  # To convert the data in numeric
    df3 = df3.dropna().reset_index(drop=True)  # To remove NaN values from data frame
    print(df3)  # To print the data frame-3
    df3.to_excel('df3.xlsx')  # To save the data frame-3
    df3.plot(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], figsize = [18,8])  # To plot line graph       
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('CO2 emissions (metric tons per capita)', fontsize = 15)  # To label the y-axis
    plt.title('CO2 emissions (metric tons per capita)', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15)  # To increase the xtick font size
    plt.rc('ytick', labelsize = 15)  # To increase the ytick font size
    plt.grid()  # To show grid on graph
    plt.savefig('line_graph-1.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph


def line_graph2():  # Define a function for line graph to show trend of Urban population for the selected countries
    df4 = df[df['Indicator Name'].isin(['Population, total'])].reset_index(drop=True)  # To create data frame with required indicator 
    df4 = df4[df4['Country Name'].isin(['Australia','Brazil','Canada','China','France',
                                        'Japan','Italy','India','Germany'])].reset_index(drop=True)  # To filter the data frame with required counties
    df4 = df4.T.reset_index(drop=False)  # To transpose the data
    df4 = df4.rename(columns=df4.iloc[0])  # To set the columns name
    df4 = df4.iloc[2:].reset_index(drop=True)  # To remove the first two rows
    df4.rename({'Country Name': 'Year'}, axis=1, inplace=True)  # To remane the first column of data frame
    df4['Year'] = pd.to_numeric(df4['Year'])  # To convert the data in numeric
    df4 = df4.dropna().reset_index(drop=True)  # To remove NaN values from data frame
    print(df4)  # To print the data frame-4
    df4.to_excel('df4.xlsx')  # To save the data frame-4
    df4.plot(x = 'Year', y = ['Australia','Brazil','Canada','China','France','Japan','Italy','India','Germany'], figsize = [18,8])  # To plot line graph
    plt.xlim(1990, 2020)   # To set the limit on x-axis        
    plt.xlabel('Year', fontsize = 15)  # To label the x-axis
    plt.ylabel('Population, total', fontsize = 15)  # To label the y-axis
    plt.title('Population, total', size = 18, color = 'black')  # To give title of graph
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize = 15, loc = 'upper left')  # To show the legends
    plt.rc('xtick', labelsize = 15) # To increase the xtick font size
    plt.rc('ytick', labelsize = 15) # To increase the ytick font size
    plt.grid()  # To show grid on graph
    plt.savefig('line_graph-2.jpeg', dpi = 300)  # To save the graph
    plt.show()  # To show the graph
    
    
bar_graph1()
bar_graph2()
line_graph1()
line_graph2()


