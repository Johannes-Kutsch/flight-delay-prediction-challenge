RED = "\033[38;5;196m"
RED_L = "\033[38;5;1m"
YEL = "\033[38;5;226m"
YEL_L = "\033[38;5;11m"
GRE = "\033[38;5;76m"
GRE_L = "\033[38;5;2m"
BLU_L = "\033[94m"
PUR = "\033[95m"
ORA = "\033[38;5;208m"
ORA_L = "\033[38;5;9m"
RES = "\033[0m"

def print_color_codes():
    for code in range(0, 356):
        print(f"\033[38;5;{code}m{code:3}\033[0m", end=" ")

CMAPS = ['Blues', 'autumn', 'bone_r', 'coolwarm', 'Pastel1', 'Pastel1_r', 'Set2']
PLOT_COLORS = ['#7E57C2', "#296B64FF"]

PLOT_TITLE_FONTSIZE = 14
TITLE_FONTSIZE = 24