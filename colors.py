import parameters as p


class Colorer():
    """
    Defined by ANSI escape sequences.
    See https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    and https://stackoverflow.com/questions/287871/how-do-i-print-color-text-to-the-terminal
    """
    _base = '\033[38;2;{};{};{}m'  # ;R;G;B
    _endc = '\033[0m'

    def __init__(self, name, path) -> None:
        self.name = name
        self.load_palette(path)

    def load_palette(self, path):
        """
        Reads csv and sets self.colors as dict {color_name: (R, G, B)}
        """
        with open(path, 'r') as r:
            lines = r.readlines()
        for a in range(len(lines)):
            lines[a] = lines[a].split(';')
            lines[a][1] = lines[a][1].split(',')
            lines[a][1] = [int(lines[a][1][b]) for b in range(3)]
        self.colors = {line[0]: tuple(line[1]) for line in lines}

    def color(self, txt, target_color) -> str:
        """
        Takes text str and palette color name str. Returns colored text str.
        """
        target_color = Colorer._base.format(*self.colors[target_color])
        return target_color + txt + Colorer._endc

    def color_rgb(self, txt, r, g, b) -> str:
        """
        Takes text str and color r g b. Returns colored text str.
        """
        target_color = Colorer._base.format(r, g, b)
        return target_color + txt + Colorer._endc

    def palette_showcase(self) -> None:
        """
        Lists available colors in terminal with their names and colored █
        """
        print(self.name)
        width = max([len(color) for color in self.colors])
        a = 1
        for color in self.colors:
            spaces = width - len(color) + 2
            print(f'[{a}] {color}' + spaces * ' ' + f'{self.color("█", color)}')
            a += 1


if __name__ == '__main__':
    cube9 = Colorer(p.NAME_CUBE9, p.PATH_CUBE9)
    rainbow = Colorer(p.NAME_RAINBOW, p.PATH_RAINBOW)
    cube9.palette_showcase()
    print('')
    rainbow.palette_showcase()
    print(rainbow.color('Orange example', 'orange'))
    print(rainbow.color_rgb('Teal example', 0, 127, 127))
