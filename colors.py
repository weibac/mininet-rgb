from abc import ABC


class Colorer(ABC):
    """
    Defined by ANSI escape sequences.
    See https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    and https://stackoverflow.com/questions/287871/how-do-i-print-color-text-to-the-terminal
    """
    _base = '\033[38;2;{};{};{}m'  # ;R;G;B
    _endc = '\033[0m'

    def __init__(self) -> None:
        self.colors = None

    def color(self, txt, target_color) -> str:
        target_color = Colorer._base.format(*self.colors[target_color])
        return target_color + txt + Colorer._endc

    def palette_showcase(self) -> None:
        width = max([len(color) for color in self.colors])
        for color in self.colors:
            spaces = width - len(color) + 2
            print(f'{color}' + spaces * ' ' + f'{self.color("█", color)}')


class Cube9(Colorer):
    """
    8 vertices of RGB cube + its center
    """
    def __init__(self) -> None:
        super().__init__()
        self.colors = {
            'black': (0, 0, 0),
            'grey': (127, 127, 127),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'magenta': (255, 0, 255),
            'cyan': (0, 255, 255)
        }


class RainbowBGW(Colorer):
    """
    ROYGBV (not I) + Black, Grey, White
    """
    def __init__(self) -> None:
        super().__init__()
        self.colors = {
            'black': (0, 0, 0),
            'grey': (127, 127, 127),
            'white': (255, 255, 255),
            'red': (255, 0, 0),
            'orange': (255, 127, 0),
            'yellow': (255, 255, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'violet': (127, 0, 255)
        }


cube9 = Cube9()  # TODO: figure out why can't call class directly
rainbow = RainbowBGW()
cube9.palette_showcase()
print('')
rainbow.palette_showcase()
