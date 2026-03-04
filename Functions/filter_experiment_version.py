import numpy as np

def filter_experiment_version(dataframes, version_name):
    """
        GOAL: Filter multiple dataframes to only include rows corresponding to a specific experiment version.
        INPUTS:
            dataframes: dictionary of pandas dataframes to filter
            version_name: string indicating which experiment version to keep    
    """

    # check that requested version name is acceptable
    acceptable_inputs = ("all", 
                        "cd1_2025_click_desired_1_identify_best_1",
                        "cd1_2025_click_desired_0_identify_best_1", 
                        "cd1_2025_click_desired_1_identify_best_0", 
                        "cd1_2025_click_desired_0_identify_best_0",
                        "cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80",
                        "cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80",
                        "versions_equal_difficulty_across_gain_loss",
                        "versions_asymmetric_difficulty_across_gain_loss",
                        "versions_equal_and_asymmetric_difficulty_click_desired_1")
    # check current version name is in acceptable inputs, print requested name if not
    assert version_name in acceptable_inputs, f"Error: version_name '{version_name}' is not an acceptable input. Acceptable inputs are: {acceptable_inputs}"
    
    # sanity checks
    # check type of column is compatible with using .unique() function
    # get name of first dataframe in dictionary
    df_name = list(dataframes.keys())[0]
    assert type(dataframes[df_name]['exp_ID'].unique()) == np.ndarray, "Error: exp_ID column should be of type numpy.ndarray in order to use .unique()"
    # then check that there are 4 unique experiment IDs
    # CONSTANCE assert dataframes[df_name]['exp_ID'].nunique() == 4, "Error: There should be 4 unique experiment IDs."

    # filter dataframes based on requested version name


    if version_name == 'all':
        # if version name is 'all', return the dataframes as is
        return dataframes
    
    elif version_name == 'versions_equal_difficulty_across_gain_loss':
        for current_df_name,current_df in dataframes.items():
            # only keep rows with current exp versions
            dataframes[current_df_name] = current_df[(current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_1') | 
                                                     (current_df['exp_ID'] == 'cd1_2025_click_desired_0_identify_best_1') | 
                                                     (current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_0') | 
                                                     (current_df['exp_ID'] == 'cd1_2025_click_desired_0_identify_best_0')] 

            # throw error if input version_name is not present in dataset
            if dataframes[current_df_name].empty:
                raise ValueError("Requested version is not present in the dataset")
            
        # after going through each dataframe, return the filtered dataframes variable
        return dataframes
    
    elif version_name == 'versions_asymmetric_difficulty_across_gain_loss':
        for current_df_name,current_df in dataframes.items():
            # only keep rows with current exp versions
            dataframes[current_df_name] = current_df[(current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80') | 
                                                     (current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80')] 

            # throw error if input version_name is not present in dataset
            if dataframes[current_df_name].empty:
                raise ValueError("Requested version is not present in the dataset")
            
        # after going through each dataframe, return the filtered dataframes variable
        return dataframes
    
    
    elif version_name == 'versions_equal_and_asymmetric_difficulty_click_desired_1':
        # only keep rows with current exp versions
        for current_df_name,current_df in dataframes.items():
            dataframes[current_df_name] = current_df[(current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_1') | 
                                                     (current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_0') | 
                                                     (current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80') | 
                                                     (current_df['exp_ID'] == 'cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80')] 

            # throw error if input version_name is not present in dataset
            if dataframes[current_df_name].empty:
                raise ValueError("Requested version is not present in the dataset")
    
        # after going through each dataframe, return the filtered dataframes variable
        return dataframes
    
    else: # return single version
        for current_df_name,current_df in dataframes.items():
            # only keep rows with current exp version
            dataframes[current_df_name] = current_df[current_df['exp_ID'] == version_name] 

            # throw error if input version_name is not present in dataset
            if dataframes[current_df_name].empty:
                raise ValueError("Requested version is not present in the dataset")
            
            # throw error if other exp_IDs are still present - and display which ones
            # assert dataframes[current_df_name]['exp_ID'].unique() == version_name, "Error: Other exp_IDs are still present in filtered dataframe. Present exp_IDs are: \n " + str(dataframes[df_name]['exp_ID'].unique())

        # after going through each dataframe, return the filtered dataframes variable
        return dataframes