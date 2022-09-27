class AnColors:
    """
    Defined by ANSI escape sequences.
    See https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    and https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
    _base = '\033[38;2;{};{};{}m'
    _endc = '\033[0m'
    _colors = {
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'magenta': (255, 0, 255),
        'cyan': (0, 255, 255),
        'grey': (127, 127, 127),
        'white': (255, 255, 255)
    }

    def printc(txt, color):
        color = AnColors._base.format(*AnColors._colors[color])
        print(color + txt + AnColors._endc)


AnColors.printc('Red', 'red')
