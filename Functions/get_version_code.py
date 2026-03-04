import re

def get_version_code(version_name_long):
    '''

    GOAL: Map long version names to short version codes.
    INPUT:
        version_name_long: string indicating the long version name
    OUTPUT:
        version_code: string indicating the short version code
    '''


    match version_name_long:
        case "all":
            version_code = "all"
        case "cd1_2025_click_desired_1_identify_best_1":
            version_code = "v11"
        case "cd1_2025_click_desired_0_identify_best_1":
            version_code = "v01"
        case "cd1_2025_click_desired_1_identify_best_0":
            version_code = "v10"
        case "cd1_2025_click_desired_0_identify_best_0":
            version_code = "v00"
        case "cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80":
            version_code = "v11_diff70_80"
        case "cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80":
            version_code = "v10_diff70_80"
        case "versions_equal_difficulty_across_gain_loss":
            version_code = "v_equal_difficulty"
        case "versions_asymmetric_difficulty_across_gain_loss":
            version_code = "v_asymmetric_difficulty"
        case "versions_equal_and_asymmetric_difficulty_click_desired_1":
            version_code = "v_equal_and_asymmetric_difficulty"  
        case _:
            raise ValueError("Error: Invalid requested version_name.") 

    # Return the version code
    return version_code


    ''' 
    """
    Extract two numerical values from exp_ID string, e.g. cd1_2025_click_desired_1_identify_best_1 --> v_1_1
    INPUT:
    df: pandas dataframe with exp_ID column

    OUTPUT:
    string: version code

    NOTES:
    - assumes all rows have the same exp_ID format
    - only to get version of a single experiment version
    """

    exp_id = df['exp_ID'].unique()  # Assuming all rows have the same exp_ID format
    
    if len(exp_id) == 1:

        # Get the unique exp_ID string
        exp_id = exp_id[0]  
        
        # initialise
        version_code = ""

        # Extract the two numerical values from the exp_ID string (the first value follows "click_desired_" and the second value follows "identify_best_")
        match = re.search(r"click_desired_(\d+).*identify_best_(\d+)(?:_(.*))?", exp_id)
        if match:
            click_desired = int(match.group(1))
            identify_best = int(match.group(2))
            suffix        = match.group(3) or ""   # (or "" if nothing there)

            # Create the version code string
            version_code = f"v{click_desired}{identify_best}{'_' + suffix if suffix else ''}"
        else:
            # Handle cases where the exp_ID format might be different
            raise KeyError(f"Warning: Could not parse exp_ID '{exp_id}'. Using default values.")
        
    else:
       # If there are multiple unique exp_IDs, set version_code to "v_all"
       print("Multiple or no unique exp_IDs found in the dataframe.")
       version_code = "vALL"

    # Return the version code
    return version_code

    '''