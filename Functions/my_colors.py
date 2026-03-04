import numpy as np

def load_my_colors():
    # create dictionary to hold colors
    # https://rgbacolorpicker.com/hex-to-rgba
    # https://www.w3schools.com/colors/colors_rgb.asp
    scale = 255 # to convert RGB 0-255 to matplotlib 0-1
    c = { 
        "light_purple" : np.divide([180, 128, 180], scale),
        "medium_purple" : np.divide([172, 47, 172], scale), 

        "light_turquoise" : np.divide([126, 196, 196], scale), 
        "medium_turquoise" : np.divide([47, 150, 150], scale), 

        "light_green" : np.divide([119, 211, 119], scale), 
        "medium_green" : np.divide([34, 139, 34], scale), 
        "dark_green" : np.divide([32, 89, 11], scale),  #np.divide([54, 112, 32], scale),
        "very_dark_green" : np.divide([0, 60, 0], scale),

        "light_red": np.divide([205, 110, 110], scale), 
        "medium_red": np.divide([205, 92, 92], scale), 
        "dark_red": np.divide([159, 49, 49], scale), 

        "light_gray": np.divide([211,211,211], scale), 
        "light_medium_gray": np.divide([180, 180, 180], scale),
        "medium_gray": np.divide([140, 140, 140], scale), 
        "dark_gray": np.divide([80, 80, 80], scale), 

        "light_orange": np.divide([251, 235, 217], scale), 
        "light_brown": np.divide([221, 210, 198], scale),
        "light_blue": np.divide([220, 232, 244], scale),

    }
    return c