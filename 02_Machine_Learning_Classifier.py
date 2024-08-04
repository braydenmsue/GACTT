import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

import seaborn as sns
import matplotlib.pyplot as plt


# if drinking location is missing, impute based on other section values
def impute_locations(row):
    if pd.isna(row['Where do you typically drink coffee?']):
        locations = []
        if pd.notna(row['How do you brew coffee at home?']):
            locations.append('At home')
        if pd.notna(row['On the go, where do you typically purchase coffee?']):
            locations.append('On the go')
        if locations:
            return ', '.join(locations)
    return row['Where do you typically drink coffee?']


# Fill in boolean options associated with drinking locations column
def fill_location_bools(row):
    options = ['At home',
               'At the office',
               'On the go',
               'At a cafe',
               'None of these'
               ]

    person_locations = row['Where do you typically drink coffee?'].split(', ')

    for option in options:
        column_name = f'Where do you typically drink coffee? ({option})'
        if pd.isna(row[column_name]):
            if option in person_locations:
                row[column_name] = 'TRUE'
            else:
                row[column_name] = 'FALSE'

    return row


# (Outlier case) if there are values in boolean (all false) but nothing in the main column, remove the boolean values
def clean_brew_method(row):
    options = ['Pour over',
               'French press',
               'Espresso',
               'Coffee brewing machine (e.g. Mr. Coffee)',
               'Pod/capsule machine (e.g. Keurig/Nespresso)',
               'Instant coffee',
               'Bean-to-cup machine',
               'Cold brew',
               'Coffee extract (e.g. Cometeer)',
               'Other'
               ]

    person_methods = pd.isna(row['How do you brew coffee at home?'])
    for option in options:
        column_name = f'How do you brew coffee at home? ({option})'
        if person_methods:
            row[column_name] = np.nan

    return row


def encode_categories(df):
    # Define mappings for age and cups of coffee
    age_mapping = {
        '<18 years old': 0,
        '18-24 years old': 1,
        '25-34 years old': 2,
        '35-44 years old': 3,
        '45-54 years old': 4,
        '55-64 years old': 5,
        '>65 years old': 6
    }

    coffee_mapping = {
        'Less than 1': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        'More than 4': 5
    }

    # Apply mappings to the DataFrame
    df.loc[:, 'What is your age?'] = df['What is your age?'].map(age_mapping)
    df.loc[:, 'How many cups of coffee do you typically drink per day?'] = df[
        'How many cups of coffee do you typically drink per day?'].map(coffee_mapping)

    return df


def generate_tables(df):
    # Drop rows with missing values in required columns
    drinkers_required_columns = ['Submission ID',
                                 'What is your age?',
                                 'How many cups of coffee do you typically drink per day?',
                                 ]

    coffee_drinkers = df.dropna(subset=drinkers_required_columns)

    # Imputation
    coffee_drinkers.loc[:, 'Where do you typically drink coffee?'] = coffee_drinkers.apply(impute_locations, axis=1)
    coffee_drinkers = coffee_drinkers.dropna(subset=['Where do you typically drink coffee?'])

    coffee_drinkers = coffee_drinkers.apply(fill_location_bools, axis=1)
    coffee_drinkers = coffee_drinkers.apply(clean_brew_method, axis=1)

    # expertise level --> mean of values
    coffee_drinkers.loc[:, 'Lastly, how would you rate your own coffee expertise?'] = \
        coffee_drinkers['Lastly, how would you rate your own coffee expertise?'].fillna(
            coffee_drinkers['Lastly, how would you rate your own coffee expertise?'].mean()
        )

    coffee_drinkers = encode_categories(coffee_drinkers)

    coffee_drinkers = coffee_drinkers.drop_duplicates()
    coffee_drinkers = coffee_drinkers.reset_index(drop=True)

    testers_required_columns = []
    # Create list of columns for test rating values
    for subject in ['A', 'B', 'C', 'D']:
        bitterness = f'Coffee {subject} - Bitterness'
        acidity = f'Coffee {subject} - Acidity'
        personal_preference = f'Coffee {subject} - Personal Preference'

        testers_required_columns.append(bitterness)
        testers_required_columns.append(acidity)
        testers_required_columns.append(personal_preference)

    coffee_testers = coffee_drinkers.dropna(subset=testers_required_columns)

    coffee_testers = coffee_testers.drop_duplicates()
    coffee_testers = coffee_testers.reset_index(drop=True)

    return coffee_drinkers, coffee_testers


def run_classification(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    # StandardScaler does nothing, data spread is very small
    # model = make_pipeline(StandardScaler(),
    #                       # MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000, activation='relu',
    #                       #               random_state=0)
    #                       GradientBoostingClassifier(n_estimators=110, learning_rate=0.3, max_depth=2, random_state=0)
    #                       )

    model = VotingClassifier([
        ('nb', GaussianNB()),
        ('knn', KNeighborsClassifier(5)),
        ('tree1', DecisionTreeClassifier(max_depth=5)),
        ('tree2', DecisionTreeClassifier(min_samples_leaf=10)),
        ('gb', GradientBoostingClassifier(n_estimators=110, learning_rate=0.3, max_depth=4, random_state=0)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
        # decreases accuracy score
        # ('mlp', MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=0))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_valid, y_pred, labels=df[target_col].unique())
    conf_matrix_df = pd.DataFrame(conf_matrix, index=df[target_col].unique(), columns=df[target_col].unique())

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('output/figures/Confusion_Matrix')
    plt.close()

    print("Classification Report:")
    print(classification_report(y_valid, y_pred))
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Accuracy: {accuracy}")

    return model, X_valid, y_valid, y_pred


def process_categorical(value):
    return 1 if value == 'TRUE' else 0


# outputs dataframe containing organized p-values for ratings between 2 groups
def mann_whitney_test(df1, df2):
    pvals = {}
    for subject in ['A', 'B', 'C', 'D']:
        values = []
        for metric in ['Bitterness', 'Acidity', 'Personal Preference']:
            za = df1[f'Coffee {subject} - {metric}']
            zb = df2[f'Coffee {subject} - {metric}']
            mwu = stats.mannwhitneyu(za, zb)
            values.append(mwu.pvalue)    # will be in same order as for-loop
        pvals[subject] = values

    return pd.DataFrame(pvals, index=['Bitterness', 'Acidity', 'Personal Preference'])

def chi_square_test(df, col1, col2):
    contingency = pd.crosstab(df[col1], df[col2])
    _, p, _, _ = stats.chi2_contingency(contingency)
    return p

def main():
    df = pd.read_csv("GACTT_RESULTS_ANONYMIZED_v2.csv")

    coffee_drinkers, coffee_testers = generate_tables(df)

    target_col = 'Lastly, what was your favorite overall coffee?'
    # categorical_features = ['Where do you typically drink coffee? (At home)',
    #                         'Where do you typically drink coffee? (At the office)',
    #                         'Where do you typically drink coffee? (On the go)',
    #                         'Where do you typically drink coffee? (At a cafe)',
    #                         'Where do you typically drink coffee? (None of these)',
    #                         'How do you brew coffee at home? (Pour over)',
    #                         'How do you brew coffee at home? (French press)',
    #                         'How do you brew coffee at home? (Espresso)',
    #                         'How do you brew coffee at home? (Coffee brewing machine (e.g. Mr. Coffee))',
    #                         'How do you brew coffee at home? (Pod/capsule machine (e.g. Keurig/Nespresso))',
    #                         'How do you brew coffee at home? (Instant coffee)',
    #                         'How do you brew coffee at home? (Bean-to-cup machine)',
    #                         'How do you brew coffee at home? (Cold brew)',
    #                         'How do you brew coffee at home? (Coffee extract (e.g. Cometeer))',
    #                         'How do you brew coffee at home? (Other)'
    #                         ]

    feature_cols = ['What is your age?',
                    'How many cups of coffee do you typically drink per day?',
                    # Decreases accuracy score, probably due to removal of records with NaN value
                    # 'Lastly, how would you rate your own coffee expertise?'
                    ]   # + categorical_features

    for subject in ['A', 'B', 'C', 'D']:
        bitterness = f'Coffee {subject} - Bitterness'
        acidity = f'Coffee {subject} - Acidity'
        personal_preference = f'Coffee {subject} - Personal Preference'
        feature_cols += [bitterness, acidity, personal_preference]

    # --------------------------------------------------------------------------------------------------------------
    # Statistical Tests (Non-Parametric)
    # --------------------------------------------------------------------------------------------------------------

    # Observations are independent; values (scores) are ordinal --> Mann-Whitney U Test
    stats_df = coffee_testers.dropna(subset=[target_col]+feature_cols)

    young_testers = stats_df[stats_df['What is your age?'] < 3]
    old_testers = stats_df[stats_df['What is your age?'] >= 3]

    r_age = mann_whitney_test(young_testers, old_testers)

    stats_gender = stats_df.dropna(subset=['Gender'])
    male_testers = stats_gender[stats_gender['Gender'] == 'Male']
    female_testers = stats_gender[stats_gender['Gender'] == 'Female']
    other_testers = stats_gender[stats_gender['Gender'].isin(['Non-binary', 'Prefer not to say', 'Other (please specify)'])]

    r_male_female = mann_whitney_test(male_testers, female_testers)
    r_male_other = mann_whitney_test(male_testers, other_testers)
    r_female_other = mann_whitney_test(female_testers, other_testers)

    stats_expertise = stats_df.dropna(subset=['Lastly, how would you rate your own coffee expertise?'])
    low_expertise = stats_expertise[stats_expertise['Lastly, how would you rate your own coffee expertise?'] <= 5]
    high_expertise = stats_expertise[stats_expertise['Lastly, how would you rate your own coffee expertise?'] > 5]

    r_expertise = mann_whitney_test(low_expertise, high_expertise)

    rename_mapping = {
        'What is your age?': 'Age',
        'Gender': 'Gender',
        'Lastly, how would you rate your own coffee expertise?': 'Expertise',
        'Lastly, what was your favorite overall coffee?': 'Coffee',
        'What is your favorite coffee drink?': 'Drink'
    }
    stats_chi = stats_df.rename(columns=rename_mapping)
    factors = ['Age', 'Gender', 'Expertise']
    stats_chi_coffee = stats_chi.dropna(subset=['Coffee'] + factors)

    pvals_chi_coffee = {}
    for factor in factors:
        p = chi_square_test(stats_chi_coffee, 'Coffee', factor)
        pvals_chi_coffee[factor] = p

    coffee_demographics = pd.DataFrame(pvals_chi_coffee, index=stats_chi['Coffee'].unique())

    stats_chi_method = stats_chi.dropna(subset=['Drink'] + factors)
    pvals_chi_method = {}
    for factor in factors:
        p = chi_square_test(stats_chi, 'Drink', factor)
        pvals_chi_method[factor] = p

    drink_demographics = pd.DataFrame(pvals_chi_method, index=stats_chi['Drink'].unique())

    output_dir = "output/stat_tests"
    os.makedirs(output_dir, exist_ok=True)

    r_age.to_csv(os.path.join(output_dir, 'rating_age.csv'), index=False)
    r_male_female.to_csv(os.path.join(output_dir, 'rating_male_female.csv'), index=False)
    r_male_other.to_csv(os.path.join(output_dir, 'rating_male_other.csv'), index=False)
    r_female_other.to_csv(os.path.join(output_dir, 'rating_female_other.csv'), index=False)
    r_expertise.to_csv(os.path.join(output_dir, 'rating_expertise.csv'), index=False)

    coffee_demographics.to_csv(os.path.join(output_dir, 'chi_coffee_demographics.csv'), index=False)
    drink_demographics.to_csv(os.path.join(output_dir, 'chi_drink_demographics.csv'), index=False)

    mwu_titles = {
        'Young vs Old': r_age,
        'Male vs Female': r_male_female,
        'Male vs Misc Gender': r_male_other,
        'Female vs Misc Gender': r_female_other,
        'Low vs High Expertise': r_expertise
    }

    chi_titles = {
        'Favorite coffee vs Demographic': coffee_demographics,
        'Favorite drink vs Demographic': drink_demographics
    }

    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write('P-VALUES RESULTING FROM TESTS \n')
        f.write('   - Rounded to 5 decimal places\n\n')
        f.write('----------------------------------------------------------------------\n')
        f.write('MANN-WHITNEY U\n')
        for title, df in mwu_titles.items():

            f.write('----------------------------------------------------------------------\n')
            f.write(f'{title}\n')
            f.write(df.round(5).to_string())
            f.write('\n')
        f.write('----------------------------------------------------------------------\n\n')
        f.write('----------------------------------------------------------------------\n')
        f.write('CHI-SQUARE\n')
        for title, df in chi_titles.items():
            f.write('----------------------------------------------------------------------\n')
            f.write(f'{title}\n')
            f.write(df.round(5).to_string())
            f.write('\n\n')

    # --------------------------------------------------------------------------------------------------------------
    # ML (Classification)
    #   Commented out code is for classification to predict favourite drink type
    # --------------------------------------------------------------------------------------------------------------

    # partitions for classifying favourite drink type; typed listed from highest to lowest count
    # group_assignments = {
    #     'Pourover': 1,
    #     'Latte': 2,
    #     'Regular drip coffee': 2,
    #     'Other': 2,
    #     'Cappuccino': 3,
    #     'Espresso': 3,
    #     'Cortado': 3,
    #     'Americano': 3,
    #     'Blended drink (e.g. Frappuccino)': 3,
    #     'Mocha': 3,
    #     'Cold brew': 2,
    #     'Iced coffee': 1
    # }
    # 3 groups, each with approximately 1/3 of the records
    # group_names = {
    #     1: 'Basic Drinks',
    #     2: 'Intermediate Drinks',
    #     3: 'Specialty Drinks'
    # }

    # coffee_drinkers_ML['group'] = coffee_drinkers_ML[target_col].map(group_assignments)
    # coffee_drinkers_ML['group_name'] = coffee_drinkers_ML['group'].map(group_names)

    # for col in categorical_features:
    #     coffee_drinkers_ML.loc[:, col] = coffee_drinkers_ML[col].apply(process_categorical)
    # --------------------------------------------------------------------------------------------------------------

    coffee_drinkers_ML = coffee_drinkers.dropna(subset=[target_col] + feature_cols)
    coffee_drinkers_ML.to_csv('output/drinkerML.csv', index=False)
    model, X_valid, y_valid, y_pred = run_classification(coffee_drinkers_ML, target_col, feature_cols)
    # print(coffee_drinkers_ML[target_col].value_counts())


    coffee_drinkers.to_csv('output/drinker.csv', index=False)
    coffee_testers.to_csv('output/tester.csv', index=False)


if __name__ == '__main__':
    main()
