class log_font_colors:
    OKGREEN = "\033[92m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


def printColored(value):
    print(f"{log_font_colors.OKGREEN}{value}{log_font_colors.ENDC}")
