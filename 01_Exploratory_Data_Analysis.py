# The Great American Coffee Taste Test

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

question_map = {
    'Gender': 'Gender',
    'What is your age?': 'Age',
    'Education Level': 'Education',
    'Do you work from home or in person?': 'Workplace',
    'Ethnicity/Race': 'Ethnicity',
    'Employment Status': 'Employment',
    'Number of Children': 'Children',
    'Political Affiliation': 'Politics',
    'How many cups of coffee do you typically drink per day?': 'Cups per day',
    'What is your favorite coffee drink?': 'Favorite coffee drink',
    'Before today\'s tasting, which of the following best described what kind of coffee you like?': 'Coffee preference',
    'How strong do you like your coffee?': 'Strength',
    'What roast level of coffee do you prefer?': 'Roast level',
    'How much caffeine do you like in your coffee?': 'Caffeine level',
    'Lastly, how would you rate your own coffee expertise?': 'Expertise level',
    'Do you like the taste of coffee?': 'Like coffee',
    'Do you know where your coffee comes from?': 'Know coffee origions',
    'What is the most you\'ve ever paid for a cup of coffee?': 'Max ever paid for a coffee cup',
    'What is the most you\'d ever be willing to pay for a cup of coffee?': 'Max willing to pay for a coffee cup',
    'Do you feel like you’re getting good value for your money when you buy coffee at a cafe?': 'Good value for money spent at cafe',
    'Approximately how much have you spent on coffee equipment in the past 5 years?': 'Equipment cost',
    'Do you feel like you’re getting good value for your money with regards to your coffee equipment?': 'Equipment value',
    'Between Coffee A, Coffee B, and Coffee C which did you prefer?': 'A or B or C',
    'Between Coffee A and Coffee D, which did you prefer?': 'A or D',
    'Lastly, what was your favorite overall coffee?': 'Favorite coffee'
}

# Define dictionary for coffee drinking locations
where_dict = {
    'At home': 'Where do you typically drink coffee? (At home)',
    'At office': 'Where do you typically drink coffee? (At the office)',
    'On the go': 'Where do you typically drink coffee? (On the go)',
    'Drink at a cafe': 'Where do you typically drink coffee? (At a cafe)',
    'Others': 'Where do you typically drink coffee? (None of these)'
}

# Define dictionary for coffee brewing methods
how_dict = {
    'Pour over': 'How do you brew coffee at home? (Pour over)',
    'French press': 'How do you brew coffee at home? (French press)',
    'Espresso': 'How do you brew coffee at home? (Espresso)',
    'Coffee brewing machine': 'How do you brew coffee at home? (Coffee brewing machine (e.g. Mr. Coffee))',
    'Pod/capsule machine': 'How do you brew coffee at home? (Pod/capsule machine (e.g. Keurig/Nespresso))',
    'Instant coffee': 'How do you brew coffee at home? (Instant coffee)',
    'Bean-to-cup machine': 'How do you brew coffee at home? (Bean-to-cup machine)',
    'Cold brew': 'How do you brew coffee at home? (Cold brew)',
    'Coffee Extract': 'How do you brew coffee at home? (Coffee extract (e.g. Cometeer))',
    'Others': 'How do you brew coffee at home? (Other)'
}

# Define dictionary for coffee purchasing locations
buy_where_dict = {
    'National chain': 'On the go, where do you typically purchase coffee? (National chain (e.g. Starbucks, Dunkin))',
    'Local café': 'On the go, where do you typically purchase coffee? (Local cafe)',
    'Drive-thru': 'On the go, where do you typically purchase coffee? (Drive-thru)',
    'Specialty coffee shop': 'On the go, where do you typically purchase coffee? (Specialty coffee shop)',
    'Deli or supermarket': 'On the go, where do you typically purchase coffee? (Deli or supermarket)',
    'Others': 'On the go, where do you typically purchase coffee? (Other)'
}

    # Define dictionary for coffee additives
add_dict = {
    'Just black': 'Do you usually add anything to your coffee? (No - just black)',
    'Add Milk, dairy alternative, or coffee creamer': 'Do you usually add anything to your coffee? (Milk, dairy alternative, or coffee creamer)',
    'Add Sugar or sweetener': 'Do you usually add anything to your coffee? (Sugar or sweetener)',
    'Add flavor syrup': 'Do you usually add anything to your coffee? (Flavor syrup)',
    'Add Others': 'Do you usually add anything to your coffee? (Other)'
}

# Define dictionary for types of dairy added to coffee
dairy_dict = {
    'Whole milk': 'What kind of dairy do you add? (Whole milk)',
    'Skim milk': 'What kind of dairy do you add? (Skim milk)',
    'Half and half': 'What kind of dairy do you add? (Half and half)',
    'Coffee creamer': 'What kind of dairy do you add? (Coffee creamer)',
    'Flavored coffee creamer': 'What kind of dairy do you add? (Flavored coffee creamer)',
    'Oat milk': 'What kind of dairy do you add? (Oat milk)',
    'Almond milk': 'What kind of dairy do you add? (Almond milk)',
    'Soy milk': 'What kind of dairy do you add? (Soy milk)',
    'Others': 'What kind of dairy do you add? (Other)'
}

# Define dictionary for sweeteners added to coffee
sweetener_dict = {
    'Granulated Sugar': 'What kind of sugar or sweetener do you add? (Granulated Sugar)',
    'Artificial Sweeteners': 'What kind of sugar or sweetener do you add? (Artificial Sweeteners (e.g., Splenda))',
    'Honey': 'What kind of sugar or sweetener do you add? (Honey)',
    'Maple Syrup': 'What kind of sugar or sweetener do you add? (Maple Syrup)',
    'Stevia': 'What kind of sugar or sweetener do you add? (Stevia)',
    'Agave Nectar': 'What kind of sugar or sweetener do you add? (Agave Nectar)',
    'Brown Sugar': 'What kind of sugar or sweetener do you add? (Brown Sugar)',
    'Raw Sugar': 'What kind of sugar or sweetener do you add? (Raw Sugar (Turbinado))'
}

# Define dictionary for reasons for drinking coffee
reason_dict = {
    'It tastes good': 'Why do you drink coffee? (It tastes good)',
    'I need the caffeine': 'Why do you drink coffee? (I need the caffeine)',
    'I need the ritual': 'Why do you drink coffee? (I need the ritual)',
    'Coffee makes me go to the bathroom': 'Why do you drink coffee? (It makes me go to the bathroom)',
    'Others': 'Why do you drink coffee? (Other)'
}

# Helper to create bar plots
def create_bar_plot(data, x_col, y_col, title='', palette='pink', rotation=45):
    """ Creates and saves a bar plot from the given data.
    Parameters:
    data (DataFrame): The data to plot.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    title (str): The title of the plot.
    palette (str): The color palette for the plot.
    rotation (int): The rotation angle for x-axis labels.
    """
    sorted_data = data.sort_values(by=y_col, ascending=False)
    sns.barplot(data=sorted_data, x=x_col, y=y_col, hue=x_col, palette=palette, legend=False)
    plt.xticks(rotation=rotation)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    output_dir = "output/figures"
    os.makedirs(output_dir, exist_ok=True)
    # Replace spaces with underscores in the title
    sanitized_title = title.replace(' ', '_').replace('/', '_').replace('?', '_')
    plt.savefig(os.path.join(output_dir, f"{sanitized_title}.png"))
    plt.close()


def create_grouped_bar_plot(df, target_col, group_by, y_col, width=0.13, use_percentages=False):
    """
    Creates and saves a grouped bar plot from the given data.
    Parameters:
        df: data to plot
        target_col: column name for desired x-axis
        group_by: column name to group by unique items
        y_col: label for y axis
        width: width of bars in plot
        use_percentages: toggle for plots displaying raw counts or percentage of group_by group
    """
    df = df.dropna(subset=[target_col, group_by])

    x_labels = df[target_col].unique()
    group_labels = df[group_by].unique()

    grouped_data = df.groupby([target_col, group_by]).size().reset_index(name='count')
    pivot_data = grouped_data.pivot(index=target_col, columns=group_by, values='count')

    if use_percentages:
        counts_per_group = pivot_data.sum(axis=0)
        percentages = pivot_data.div(counts_per_group, axis=1) * 100
    else:
        percentages = pivot_data    # using counts instead of percentages

    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(layout='constrained')
    multiplier = 0
    for index, label in enumerate(group_labels):
        offset = width * multiplier
        rects = ax.bar(x + offset, percentages[label], width, label=label)
        ax.bar_label(rects, padding=5, fmt='%d')
        multiplier += 1

    title = f"{target_col} by {group_by}"
    if use_percentages:
        title += " Percentage"
        ax.set_ylim(0, 100)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(group_labels) - 1) / 2, x_labels)

    # if legend is long or bars are high, put it outside the plot
    if len(group_labels) > 9:
        ax.legend(title=group_by, bbox_to_anchor=(1.02, 1))
    else:
        ax.legend(title=group_by, loc='upper left', fontsize='small')

    output_dir = "output/figures"
    os.makedirs(output_dir, exist_ok=True)
    # Replace spaces with underscores in the title
    sanitized_title = title.replace(' ', '_').replace('/', '_').replace('?', '_')
    plt.savefig(os.path.join(output_dir, f"{sanitized_title}.png"))
    plt.close()


# Creates and saves a heatmap showing the percentage of people who chose both columns
def create_heatmap(df, columns, title='', threshold=0.005):
    """
    Creates and saves a heatmap showing the percentage of people who chose both columns.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    columns (list): List of column names to include in the heatmap.
    title (str): The title of the heatmap.
    threshold (float): The minimum percentage to display (default is 0.5%).
    """
    # Calculate the percentage matrix
    percentage_matrix = pd.DataFrame(index=columns, columns=columns)
    total_count = df.shape[0]
    
    for col1 in columns:
        for col2 in columns:
            count_both = df[(df[col1] == 1) & (df[col2] == 1)].shape[0]
            percentage_matrix.loc[col1, col2] = count_both / total_count if total_count > 0 else 0
    # Apply threshold
    percentage_matrix = percentage_matrix.map(lambda x: x if x >= threshold else None)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(percentage_matrix, annot=True, fmt=".2%", cmap="YlOrBr", cbar=False)
    plt.title(title)
    plt.tight_layout()
    output_dir = "output/figures"
    os.makedirs(output_dir, exist_ok=True)
    # Replace spaces with underscores in the title
    sanitized_title = title.replace(' ', '_').replace('/', '_').replace('?', '_')
    plt.savefig(os.path.join(output_dir, f"{sanitized_title}.png"))
    plt.close()

def create_heatmap_from_dict(df, column_dict, title=''):
    """
    Renames columns based on a dictionary and creates a heatmap showing the percentage of people who chose both columns.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column_dict (dict): A dictionary mapping new column names to lists of columns to rename.
    title (str): The title of the heatmap.
    threshold (float): The minimum percentage to display (default is 0.5%).
    """
    # Reverse the dictionary to map old column names to new column names
    reverse_dict = {v: k for k, v in column_dict.items()}
    
    # Rename the columns in the DataFrame
    renamed = df.rename(columns=reverse_dict)
    
    # Get the list of new column names
    new_columns = list(reverse_dict.values())

    return create_heatmap(renamed, new_columns, title=title)

# Data Overview
def data_overview(df):
    """
    Prints an overview of the DataFrame including info, missing values, and summary statistics.

    Parameters:
    df (DataFrame): The DataFrame to overview.
    """
    print("Data Info:")
    df.info()
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

# Function to create value count plots
def plot_value_counts(df, column, title_suffix=''):
    """
    Plots the value counts of a specified column.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column (str): The column to plot value counts for.
    title_suffix (str): Suffix to add to the plot title.
    """
    value_counts = df[column].value_counts().reset_index(name='count')
    create_bar_plot(value_counts, column, 'count', f'{column} Distribution{title_suffix}')

# Function to sum specific columns and create a DataFrame
def sum_columns_to_df(df, column_dict, index):
    """
    Sums specific columns and creates a DataFrame from the results.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    column_dict (dict): A dictionary mapping new column names to lists of columns to sum.
    index (range): The index for the new DataFrame.

    Returns:
    DataFrame: A DataFrame with summed columns.
    """
    summed_data = {name: df[columns].sum() for name, columns in column_dict.items()}
    summed_df = pd.DataFrame(list(summed_data.items()), columns=['Category', 'count'], index=index)
    return summed_df

# Function to create percentage columns
def add_percentage(df, count_col):
    """
    Adds a percentage column to the DataFrame based on the count column.
    
    Parameters:
    df (DataFrame): The DataFrame to modify.
    count_col (str): The column name for counts.
    """
    df['percentage'] = (df[count_col] / df[count_col].sum()).round(2)
    
# Function to calculate average ratings for coffee attributes
def calculate_average_ratings(df, label):
    """
    Calculates average ratings for coffee attributes.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    label (str): The label for the coffee attributes.
    
    Returns:
    DataFrame: A DataFrame with average ratings for coffee attributes.
    """
    selected_columns = [f'Coffee {label} - Bitterness', f'Coffee {label} - Acidity', f'Coffee {label} - Personal Preference']
    coffee_ratings = df[selected_columns].mean().round(1).reset_index()
    coffee_ratings.columns = ['attribute', 'avg rating']
    return coffee_ratings

def main():
    """
    Main function to load data, generate plots, and save results.
    """
    # Load data from CSV file
    df = pd.read_csv("GACTT_RESULTS_ANONYMIZED_v2.csv")
    
    # Map long questions to shorter titles
    df.rename(columns=question_map, inplace=True)

    # Display data overview including info, missing values, and summary statistics    
    data_overview(df)

    # Plot and save value counts for various demographic and coffee-related questions
    plot_value_counts(df, 'Gender')
    plot_value_counts(df, 'Age')
    plot_value_counts(df, 'Education')
    plot_value_counts(df, 'Workplace')
    plot_value_counts(df, 'Ethnicity')
    plot_value_counts(df, 'Employment')
    # Impute missing values for 'Children' column as 0
    # NOTE: no answer doesn't necessarily mean 0 children, but comment out if needed
    # df['Children'].fillna(0, inplace=True)
    plot_value_counts(df, 'Children')
    plot_value_counts(df, 'Politics')
    plot_value_counts(df, 'Cups per day')

    where_df = sum_columns_to_df(df, where_dict, range(5))
    create_bar_plot(where_df, 'Category', 'count', 'Coffee Drinking Locations')
    where_df.to_csv('output/Coffee_Drinking_Locations.csv')

    create_heatmap_from_dict(df, where_dict, 'Coffee Drinking Locations Matrix')

    how_df = sum_columns_to_df(df, how_dict, range(10))
    create_bar_plot(how_df, 'Category', 'count', 'At Home Brew Methods')
    how_df.to_csv('output/At_Home_Brew_Methods.csv')

    create_heatmap_from_dict(df, how_dict, "Brew Methods Matrix")

    buy_where_df = sum_columns_to_df(df, buy_where_dict, range(6))
    create_bar_plot(buy_where_df, 'Category', 'count', 'Coffee Purchasing Locations')
    buy_where_df.to_csv('output/Coffee_Purchasing_Locations.csv')

    create_heatmap_from_dict(df, buy_where_dict, 'Coffee Purchasing Locations Matrix')

    # Plot and save value counts for favorite coffee drink
    plot_value_counts(df, 'Favorite coffee drink')

    add_df = sum_columns_to_df(df, add_dict, range(5))
    create_bar_plot(add_df, 'Category', 'count', 'Coffee Additives')
    add_df.to_csv('output/Coffee_Additives.csv')

    create_heatmap_from_dict(df, add_dict, 'Coffee Additives Matrix')

    dairy_df = sum_columns_to_df(df, dairy_dict, range(9))
    create_bar_plot(dairy_df, 'Category', 'count', 'Dairy Added')
    dairy_df.to_csv('output/Dairy_Added.csv')

    sweetener_df = sum_columns_to_df(df, sweetener_dict, range(8))
    create_bar_plot(sweetener_df, 'Category', 'count', 'Sweetener Added')
    sweetener_df.to_csv('output/Sweetener_Added.csv')
    
    # Plot and save value counts for various coffee preferences and expertise
    plot_value_counts(df, 'Coffee preference')
    plot_value_counts(df, 'Strength')
    plot_value_counts(df, 'Roast level')
    plot_value_counts(df, 'Caffeine level')
    plot_value_counts(df, 'Expertise level')

    avg_expertise = df['Expertise level'].mean()
    print(f"Average Coffee Expertise: {avg_expertise.round(1)}")

    reason_df = sum_columns_to_df(df, reason_dict, range(5))
    create_bar_plot(reason_df, 'Category', 'count', 'Reasons for Drinking Coffee')
    reason_df.to_csv('output/Reasons_for_Drinking_Coffee.csv')

    create_heatmap_from_dict(df, reason_dict, 'Reasons Matrix')

    # Plot the value counts for whether people like the taste of coffee
    plot_value_counts(df, 'Like coffee')
    
    # Plot the value counts for whether people know where their coffee comes from
    plot_value_counts(df, 'Know coffee origions')

    # Calculate the maximum amount people have paid for a cup of coffee and create a percentage column
    max_paid = df['Max ever paid for a coffee cup'].value_counts().reset_index(name='count')
    add_percentage(max_paid, 'count')
    create_bar_plot(max_paid, 'Max ever paid for a coffee cup', 'count')

    # Calculate the maximum amount people are willing to pay for a cup of coffee and create a percentage column
    max_acceptable = df['Max willing to pay for a coffee cup'].value_counts().reset_index(name='count')
    add_percentage(max_acceptable, 'count')
    create_bar_plot(max_acceptable, 'Max willing to pay for a coffee cup', 'count')

    # Plot the value counts for whether people feel they are getting good value for their money when buying coffee at a cafe
    plot_value_counts(df, 'Good value for money spent at cafe')
    
    # Plot the value counts for how much people have spent on coffee equipment in the past 5 years
    plot_value_counts(df, 'Equipment cost')
    
    # Plot the value counts for whether people feel they are getting good value for their money with regards to their coffee equipment
    plot_value_counts(df, 'Equipment value')

    # Calculate average ratings for Coffee A attributes and print the DataFrame
    coffee_A = calculate_average_ratings(df, 'A')
    print(coffee_A)
    coffee_A.to_csv('output/coffee_A.csv')

    # Calculate average ratings for Coffee B attributes and print the DataFrame
    coffee_B = calculate_average_ratings(df, 'B')
    print(coffee_B)
    coffee_B.to_csv('output/coffee_B.csv')

    # Calculate average ratings for Coffee C attributes and print the DataFrame
    coffee_C = calculate_average_ratings(df, 'C')
    print(coffee_C)
    coffee_C.to_csv('output/coffee_C.csv')

    # Calculate average ratings for Coffee D attributes and print the DataFrame
    coffee_D = calculate_average_ratings(df, 'D')
    print(coffee_D)
    coffee_D.to_csv('output/coffee_D.csv')

    # Plot the value counts for the preferred coffee between Coffee A, B, and C
    plot_value_counts(df, 'A or B or C')

    # Plot the value counts for the preferred coffee between Coffee A and Coffee D
    plot_value_counts(df, 'A or D')

    # Plot the value counts for the favorite overall coffee
    plot_value_counts(df, 'Favorite coffee')

    create_grouped_bar_plot(df, 'Favorite coffee', 'Age', 'count')
    create_grouped_bar_plot(df, 'Favorite coffee', 'Gender', 'count')
    create_grouped_bar_plot(df, 'Favorite coffee', 'Expertise level', 'count', width=0.09)

    create_grouped_bar_plot(df, 'Favorite coffee', 'Age', '% of Age group', use_percentages=True)
    create_grouped_bar_plot(df, 'Favorite coffee', 'Gender', '% of Gender group', use_percentages=True)
    create_grouped_bar_plot(df, 'Favorite coffee', 'Expertise level', '% of Expertise group', width=0.09, use_percentages=True)

    create_grouped_bar_plot(df, 'Roast level', 'Age', '% of Age group', use_percentages=True)
    create_grouped_bar_plot(df, 'Roast level', 'Gender', '% of Gender group', use_percentages=True)
    create_grouped_bar_plot(df, 'Roast level', 'Expertise level', '% of Expertise group', width=0.09, use_percentages=True)

if __name__ == '__main__':
    main()
