import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def votesIntoWeightedVotes_converter(TEA_NFT_data_csv, voting_weights_data_csv):
    # Create a DataFrame
    df_w_votes = pd.DataFrame()

    # Create a DataFrame
    df_all_data = pd.DataFrame()

    # Create voting weights dictionary from csv
    df_voting_weights = pd.read_csv(voting_weights_data_csv)
    voting_weight_dict = {}
    for CodeName, Weight in zip(df_voting_weights['CodeName'], df_voting_weights['Weight']):
        voting_weight_dict[CodeName] = Weight

    # read in TEA_NFT_data csv
    df_TEA_NFT_data = pd.read_csv(TEA_NFT_data_csv)

    # create df_all_data containing weighted_votes
    df_all_data = df_TEA_NFT_data.copy(deep=True)

    def multiplier(a):
        if a == 0:
            return 0
        else:
            return a * voting_weight_dict[column_name]
    for column_name in df_TEA_NFT_data:
        if column_name in ['Unnamed: 0', 'ID']:
            pass
        else:
            df_all_data[f'w_{column_name}'] = df_all_data[column_name].apply(lambda a: multiplier(a))
    df_all_data.drop(columns=['Unnamed: 0'], inplace=True)

    # create df_weighted_NFTs
    df_weighted_NFTs = df_all_data.copy(deep=True)
    df_weighted_NFTs.drop(columns=['ID', 'NFTREP_V1', 'SPEAKER_ETHCC_PARIS23', 'STUDY_GROUP_HOST_C2_22_23',
       'ETHCC_23', 'FUND_AUTHOR', 'STUDY_GROUP_HOST_360_22',
       'STUDY_GROUP_HOST_FUND_22_23', 'SPEAKER_BARCAMP_PARIS_23',
       'BARCAMP_PARIS_23', 'TEAM_BARCAMP_PARIS_23', 'FUND_MOD_1', 'FUND_MOD_2',
       'FUND_MOD_3', 'FUND_MOD_4', 'FUND_MOD_5'], inplace=True)
    student_nftWeight = df_weighted_NFTs.sum(axis=1)

    return student_nftWeight, df_all_data


def barChart_generator(student_nftWeight, barChartTitle):
    # Calculate the mean, median, and maximum
    mean_value = student_nftWeight.mean()
    median_value = student_nftWeight.median()
    max_value = int(student_nftWeight.max()) #might be an issue here if weighted number result in floats!!
    min_value = student_nftWeight.min()

    # Calculate the halfway value between median and maximum
    dMedToMaxHalf = median_value + ((max_value - median_value) / 2)
    # Calculate the halfway line from max_value
    halfWayLine = max_value/2

    # Counts how many addresses hold how many NFTs (group by number-of-NFT-held) and sorts ascending
    counts = student_nftWeight.value_counts().sort_index()

    # Ensure all values from 1 to max_value are included to maintain a flexible but consistent x-axis
    for i in range(1, (max_value+1)):
        if i not in counts:
            counts[i] = 0

    # Sort the counts again after ensuring all values from 1 to 9 are included
    counts = counts.sort_index()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values, color='blue')

    # Adding vertical lines for mean, median, and halfway point
    plt.axvline(x=mean_value, color='purple', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(x=median_value, color='red', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')
    plt.axvline(x=dMedToMaxHalf, color='darkorange', linestyle='--', linewidth=2, label=f'dMedToMax/2: {dMedToMaxHalf:.2f}')
    plt.axvline(x=halfWayLine, color='green', linestyle='--', linewidth=2, label=f'halfway: {halfWayLine:.2f}')

    # Set the x-axis ticks to range from 1 to max_value
    plt.xticks(range(5, (max_value+5), 5)) # plt.xticks(range(1, (max_value+1)))

    # Adding a custom legend with specific text values
    # Custom labels for the legend
    custom_legend_labels = []

    # Create proxy artists for the custom legend with no color
    custom_legend_handles = [Line2D([0], [0], color='none', label=label) for label in custom_legend_labels]

    # Add the mean, median, and deltaMedianToMax-line to the legend handles
    custom_legend_handles.extend([
        Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}'),
        Line2D([0], [0], color='darkorange', linestyle='--', linewidth=2, label=f'dMedToMax/2: {dMedToMaxHalf:.2f}'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'halfway: {halfWayLine:.2f}'),
        Line2D([0], [0], color='none', linestyle='', linewidth=0, label=f'max: {max_value:.2f}'),
        Line2D([0], [0], color='none', linestyle='', linewidth=0, label=f'min: {min_value:.2f}')
    ])

    # Add the legend to the plot
    plt.legend(handles=custom_legend_handles, title='Custom Legend')

    # Add labels and title
    plt.xlabel('Weight') #Number of NFTs
    plt.ylabel('Number of Addresses')
    plt.title(f'Voting power distribution - {barChartTitle}') #How many addresses have how many NFTs?

    # Display the plot
    #plt.show()

    # Save the plot as a .png file 
    plt.savefig(saveToLocation_bar) ########## <------ UNCOMMENT TO SAVE IMAGE

    # Close the plot to free memory
    plt.close()

    print(f'mean = {mean_value}')
    print(f'median_value = {median_value}')
    print(f'max_value  = {max_value}')
    print(f'min_value = {min_value}')


def whatsYourWeight_ID(student_nftWeight, df_all_data, yourAddressOrWeight):
    for ID, weight in zip(df_all_data['ID'], student_nftWeight):
        if ID == yourAddressOrWeight:
            #return weight
            print(weight)
        elif weight == yourAddressOrWeight:
            #return ID
            print(ID)
    

def scatterPlot_generator(TEA_NFT_data_csv, student_nftWeight, scatterChartTitle):
    df_NFT_calc = pd.read_csv(TEA_NFT_data_csv)
    df_NFT_calc['sumNFT'] = df_NFT_calc.drop(columns=['Unnamed: 0', 'ID']).sum(axis=1)
    df_NFT_calc['student_nftWeight'] = student_nftWeight
    df_NFT_calc['numNFToverWeight'] = df_NFT_calc['sumNFT']/df_NFT_calc['student_nftWeight']
    df_NFT_calc['WeightoverNumNFT'] = df_NFT_calc['student_nftWeight']/df_NFT_calc['sumNFT']
    # if you change the x axis to "ID" you can use the sort below (otherwise the scatter plot auto sorts and you cannot decide on what to sort)
    df_NFT_calc.sort_values(by='student_nftWeight', ascending=True, inplace=True)
    #print(df_NFT_calc[['sumNFT','student_nftWeight', 'WeightoverNumNFT', 'numNFToverWeight']].iloc[235:270]) #check for more info

    # Sample data for demonstration
    x_data1 = df_NFT_calc['student_nftWeight']
    y_data2 = df_NFT_calc['WeightoverNumNFT']

    # Plot the points
    plt.scatter(x_data1, y_data2, color='blue', alpha=0.5)

    # Set labels and title
    plt.xlabel('Weight')
    plt.ylabel('Weight / number_of_NFTs')
    plt.title(f'Weight to NFT-quantity relationship - {scatterChartTitle}')

    # Save the plot as a .png file 
    plt.savefig(saveToLocation_scatter) ########## <------ UNCOMMENT TO SAVE IMAGE

    # Close the plot to free memory
    plt.close()

    # Show plot
    #plt.show()


def rank_n_slide(df_voter_preferences, student_nftWeight):
    #df_voter_preferences = pd.read_csv(voter_preferences_csv)
    #df_voter_preferences = df_voter_preferences.copy(deep=True)
    df_voter_preferences['voting_power'] = student_nftWeight

    def multiplier(b, c, d, e, f):
        scoreA = b * c
        scoreB = b * d
        scoreC = b * e
        scoreD = b * f
        return scoreA, scoreB, scoreC, scoreD
    df_voter_preferences[['candidate_A_score', 'candidate_B_score', 'candidate_C_score', 'candidate_D_score']] = df_voter_preferences.apply(
        lambda row: multiplier(row['voting_power'], row['candidate_A_preferences'], row['candidate_B_preferences'], row['candidate_C_preferences'], row['candidate_D_preferences']), axis=1, result_type='expand')

    candidate_A_score = sum(df_voter_preferences['candidate_A_preferences'])
    candidate_B_score = sum(df_voter_preferences['candidate_B_preferences'])
    candidate_C_score = sum(df_voter_preferences['candidate_C_preferences'])
    candidate_D_score = sum(df_voter_preferences['candidate_D_preferences'])

    candidate_score_dic = {
        "candidate_A" : candidate_A_score,
        "candidate_B" : candidate_B_score,
        "candidate_C" : candidate_C_score,
        "candidate_D" : candidate_D_score,
    }
    
    # determine winning_score
    values = candidate_score_dic.values()
    winning_score = max(candidate_score_dic.values())

    # determine winner or draw
    winnersList = []
    for candidate, score in candidate_score_dic.items():
        if winning_score == score:
            winnersList.append(candidate)

    resultList = []
    if len(winnersList) > 1:
        print("DRAW!")
        resultList.append("Draw")
    else:
        print(f'Winner = {winnersList[0]}')
        resultList.append(winnersList[0])

    for candidate, score in candidate_score_dic.items():
        print(f'{candidate} - score: {score}')

    result = resultList[0]
    #print(candidate_score_dic, result)
    return(candidate_score_dic, result)


def voting_results_barChart(results, nft_weight_setting): #must be a dictionary with 4 keys containing candidate name as key and total sum of votes won as value
    # Sample data
    barChart_data = results

    # Create the bar chart
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size
    plt.bar(barChart_data.keys(), barChart_data.values(), color='blue')

    # Add title and labels
    plt.title(f'Voting results (10,000 runs) {nft_weight_setting}')
    plt.xlabel('Candidates')
    plt.ylabel('Number of votes won')

    # Save the plot as a .png file 
    plt.savefig(f'/Users/Victor/PycharmProjects/basic-voting-calc/illus_results/simulations/{nft_weight_setting}_simulationResult_BarChart.png') ########## <------ UNCOMMENT TO SAVE IMAGE

    # Close the plot to free memory
    plt.close()

    # Display the chart
    #plt.show()

def weight_allocation_barChart(voting_weights_data_csv, barChartTitle): #must be a dictionary with 4 keys containing candidate name as key and total sum of votes won as value
    # Sample data
    df_voting_weights = pd.read_csv(voting_weights_data_csv)
    barChart_data = df_voting_weights['Weight']
    code_names = df_voting_weights['CodeName']

    # Create the bar chart
    plt.figure(figsize=(10, 6))  # Optional: Set the figure size
    plt.bar(code_names, barChart_data, color='blue')

    # Add title and labels
    plt.title(f'Weight Allocation - {barChartTitle}')
    plt.xlabel('candidates')
    plt.ylabel('allocated weight')

    plt.xticks(rotation=90)

    # Save the plot as a .png file 
    #plt.savefig(saveToLocation_bar) ########## <------ UNCOMMENT TO SAVE IMAGE

    # Close the plot to free memory
    #plt.close()

    # Display the chart
    plt.show()


###  voter_preferences_random
# Load the template CSV file
def generate_random_voter_preference_data(file_path_x):
    df_preferences = pd.read_csv(file_path_x)

    # Function to generate random values that sum to 1
    def generate_random_percentages(n):
        values = np.random.rand(n)
        values /= values.sum()
        return values

    # Apply the function to each row to fill columns 2 to 5
    for index, row in df_preferences.iterrows():
        percentages = generate_random_percentages(4)
        df_preferences.at[index, 'candidate_A_preferences'] = percentages[0]
        df_preferences.at[index, 'candidate_B_preferences'] = percentages[1]
        df_preferences.at[index, 'candidate_C_preferences'] = percentages[2]
        df_preferences.at[index, 'candidate_D_preferences'] = percentages[3]

    return df_preferences

#----------------------
barChartTitle = "dictator"
scatterChartTitle = "Angela_default"

### voter preferences data:
file_path_x = '/Users/Victor/PycharmProjects/basic-voting-calc/data/test_voting_data_csv_files/voter_preferences_template.csv'  # Update this to your file path

voter_preferences_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_voting_data_csv_files/voter_equilibrium.csv"
#voter_preferences_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_voting_data_csv_files/voter_preferences_GPTbiasedA.csv"
#voter_preferences_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_voting_data_csv_files/voter_preferences_GPTclose_A.csv"
#voter_preferences_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_voting_data_csv_files/voter_preferences_GPTrandom.csv"

### NFT weights:
#voting_weights_data_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/default_voting_weights.csv"
#voting_weights_data_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/0_unweighted_control.csv"
#voting_weights_data_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/OneLV_voting_weights.csv"
#voting_weights_data_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/GPT1_voting_weights_increase_highest.csv"
#voting_weights_data_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/GPT2_voting_weights_increase_lowest.csv"
voting_weights_data_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/dictator_voting_weights.csv"


TEA_NFT_data_csv = "/Users/Victor/PycharmProjects/basic-voting-calc/data/nft_data_may_28_2024_cleaned.csv"

### Storage location (path)
saveToLocation_bar = f'/Users/Victor/PycharmProjects/basic-voting-calc/illus_results/test_exp/{barChartTitle}_weight_allocation_BarChart.png' #add your save to location
saveToLocation_scatter = f'/Users/Victor/PycharmProjects/basic-voting-calc/illus_results/test_exp/{scatterChartTitle}_scatter_plot.png' #add your save to location


### Address/weight search
yourAddressOrWeight = '0xfcbc07905fee2d64025461b8ddb27f77f256827f'

#weight_allocation_barChart(voting_weights_data_csv, barChartTitle)

#output = votesIntoWeightedVotes_converter(TEA_NFT_data_csv, voting_weights_data_csv)
#barChart_generator(output[0], barChartTitle)
#scatterPlot_generator(TEA_NFT_data_csv, output[0], scatterChartTitle)
#rank_n_slide(voter_preferences_csv, output[0])
#voting_results_barChart(results)

#--------------------------------------

voting_weights_data_list = [
    "/Users/Victor/PycharmProjects/basic-voting-calc/data/default_voting_weights.csv",
    "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/0_unweighted_control.csv",
    "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/OneLV_voting_weights.csv",
    "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/GPT1_voting_weights_increase_highest.csv",
    "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/GPT2_voting_weights_increase_lowest.csv",
    "/Users/Victor/PycharmProjects/basic-voting-calc/data/test_weight_data_csv_files/dictator_voting_weights.csv"
]


voting_weights_data_keyName_list = ["default", "control", "OneLV", "GPT1", "GPT2", "dictator"]


# create a result dictionary to store results of each voting run
result_dic = {}
run = 0

# main logic

for weight_allocation_data_n, keyName in zip(voting_weights_data_list, voting_weights_data_keyName_list):
    output = votesIntoWeightedVotes_converter(TEA_NFT_data_csv, weight_allocation_data_n)
    dic_name = keyName
    dic_name = {}
    #df_preferences = generate_random_voter_preference_data(file_path_x) ##########????
    for testRound2 in range(10000): # 10000
        run = run + 1
        df_preferences = generate_random_voter_preference_data(file_path_x) ##########????
        #df_preferences = pd.read_csv(voter_preferences_csv)
        result = rank_n_slide(df_preferences, output[0])
        intermediate = result[1]
        dic_name[run] = intermediate
    result_dic[keyName] = dic_name


print(result_dic['default'])

for nft_weight_setting in voting_weights_data_keyName_list:
    countA = 0
    countB = 0
    countC = 0
    countD = 0

    for z in result_dic[nft_weight_setting].values():
        if z == 'candidate_A':
            countA = countA + 1
        if z == 'candidate_B':
            countB = countB + 1
        if z == 'candidate_C':
            countC = countC + 1
        if z == 'candidate_D':
            countD = countD + 1

    print(countA, countB, countC, countD)
    result_now = {'candidate_A' : countA,
                'candidate_B' : countB,
                'candidate_C' : countC,
                'candidate_D' : countD}
    print(nft_weight_setting)
    voting_results_barChart(result_now, nft_weight_setting)







"""
# Load the CSV file
file_path = './data/nft_data_may_28_2024_cleaned.csv'
df = pd.read_csv(file_path)

# Assuming all columns except 'Unnamed: 0' and 'ID' are to be summed
df['NFTs_sum'] = df.drop(columns=['Unnamed: 0', 'ID']).sum(axis=1)

# Display the first few rows to verify the new column
print(df.head())

max_value = df['NFTs_sum'].max()
min_value = df['NFTs_sum'].min()
mean_value = df['NFTs_sum'].mean()
median_value = df['NFTs_sum'].median()

print(max_value)
print(min_value)
print(mean_value)
print(median_value)

# Create the dot plot
plt.figure(figsize=(10, 6))
plt.plot(df['ID'], df['NFTs_sum'], 'o', markersize=5)
plt.xlabel('IDs')
plt.ylabel('NFTs_sum')
plt.title('Dot Plot of NFTs Sum by ID')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()
"""