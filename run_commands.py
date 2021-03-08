import os

with open('commands.txt') as f:
    for line in f:
        os.system(line)