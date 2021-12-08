import sys, random

#Function to Shuffle the rows
def shuffleRows(neLayoutName, number=1):
    file = "layouts/%s.lay" % neLayoutName
    shuffledFile = "layouts/%sShuffle%s.lay" % (neLayoutName, number)
    with open(file, 'r') as f:
        array = f.read()

    lines = array.split("\n")
    shuffledRows = lines[1:-1]
    random.shuffle(shuffledRows)
    with open(shuffledFile, 'w') as f:
        f.write("%s\n" % lines[0])
        for row in shuffledRows:
            if row != '':
                f.write("%s\n" % row)
        f.write("%s" % lines[0])

    return "%sShuffle%s" % (neLayoutName, number)

#Function to rotate the layout
def rotate(neLayoutName):
    file = "layouts/%s.lay" % neLayoutName
    newLayoutFile = "layouts/%sT.lay" % neLayoutName
    with open(file, 'r') as f:
        array = f.read()

    lines = array.split("\n")
    with open(newLayoutFile, 'w') as f:
        for j in range(len(lines[0])):
            for line in lines:
                if line != '':
                    f.write(line[j])
            if j != len(lines[0])-1:
                f.write("\n")

    return "%sT" % neLayoutName

#Function to flip the layout.
def mirror(neLayoutName):
    file = "layouts/%s.lay" % neLayoutName
    flipFile = "layouts/%sF.lay" % neLayoutName
    with open(file, 'r') as f:
        array = f.read()

    lines = array.split("\n")
    with open(flipFile, 'w') as f:
        for i in range(len(lines)):
            row = lines[i]
            for j in range(len(lines[0])-1,-1,-1):
                if row != '':
                    f.write(row[j])
            if i != len(lines)-1:
                f.write("\n")

    return "%sF" % neLayoutName

def create(layoutName):
    layoutT = rotate(layoutName)
    layoutF = mirror(layoutName)
    layoutTF = mirror(layoutT)
    layoutFT = rotate(layoutF)
    layoutFTF = mirror(layoutFT)
    layoutTFT = rotate(layoutTF)
    layoutTFTF = mirror(layoutTFT)

def names(layoutNames):
    stringOfNames = ""
    for i in layoutNames:
        create(i)
        stringOfNames += "%s-%sF-%sFT-%sFTF-%sT-%sTF-%sTFT-%sTFTF-" % (i, i, i, i, i, i, i, i)

    return stringOfNames[0:-1]

if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == "-T":
        print((rotate(args[1])))
    elif args[0] == "-S":
        print((names(args[1:])))
    elif args[0] == "-ShR":
        number = 1
        if len(args) > 2:
            number = int(args[2])
        stringOfNames = ""
        for i in range(number):
            stringOfNames += "%s-" % shuffleRows(args[1], number=i)
        #print((stringOfNames[0:-1]))