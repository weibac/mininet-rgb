color_categories = {'black', 'grey', 'gray', 'white', 'red', 'orange', 'yellow',
                    'green', 'blue', 'violet'}  # TODO: un-hardcode

with open('color_names.csv', 'r') as r:
    lines = r.readlines()

# Leave only name and RGB
lines = lines[1:]
lines = [line.split(',') for line in lines]
for a in range(len(lines)):
    lines[a] = [lines[a][0].strip('"')] + [lines[a][b] for b in range(2, 5)]

# Filter colors
filtered_colors = []
for a in range(len(lines)):
    color_name_words = [word.lower() for word in lines[a][0].split(' ')]
    for word in color_name_words:
        if word in color_categories:
            filtered_colors.append([word] + lines[a][1:])

# Write filtered dataset
lines = [';'.join([color[0], ','.join(color[1:])]) + '\n' for color in filtered_colors]
# TODO: Separate above line
with open('filtered_color_labels.csv', 'w') as w:
    w.writelines(lines)
