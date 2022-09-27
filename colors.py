from abc import ABC


class Colorer(ABC):
    """
    Defined by ANSI escape sequences.
    See https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    and https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
    _base = '\033[38;2;{};{};{}m'  # RGB
    _endc = '\033[0m'

    def __init__(self) -> None:
        self._colors = None

    def printc(self, txt, color):
        color = Colorer._base.format(*self._colors[color])
        print(color + txt + Colorer._endc)


class Cube9(Colorer):
    """
    8 vertices of RGB cube + its center
    """
    def __init__(self) -> None:
        super().__init__()
        self._colors = {
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


cube9 = Cube9()  # TODO: figure out why can't call class directly
cube9.printc('█', 'cyan')
