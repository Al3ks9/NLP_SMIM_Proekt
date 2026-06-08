in_file = 'songs.csv'
out_file = 'stripped_songs.csv'

linii = []

with open (in_file, 'r') as file:
    line = file.readline()
    while line != '':
        line = line.strip()
        print(line)
        linii.append(f'{line}\n')
        line = file.readline()

with open(out_file, 'w') as file_1:
    file_1.writelines(linii)
