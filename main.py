import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


def progress_bar(progress, total):
    percent = 100*(progress/float(total))
    bar = '■' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


size_x = 32
size_y = 32

transform = transforms.Compose([transforms.Resize((size_x, size_y)),
                                transforms.ToTensor()])
folder = datasets.ImageFolder('Data/Letters', transform=transform)

net = torch.jit.load('modelCNN.pt')
net.eval()


####################
# Defining functions
####################


def getPath(x):
    return f"Input/{x}.jpg"


# Detect objects on the bitMap
def getObjects(bitMap, height, width):
    visited = torch.clone(bitMap)
    objects = []
    for i in range(height):
        for j in range(width):
            progress_bar(i * width + j + 1, height * width)

            if not visited[i][j]:
                x = []
                y = []
                queue = []
                queue.append((i, j))

                while len(queue) > 0:
                    point = queue.pop(0)
                    x1 = point[0]
                    y1 = point[1]

                    if not visited[x1][y1]:
                        if x1 > 0:
                            queue.append((x1 - 1, y1))
                        if x1 < height - 1:
                            queue.append((x1 + 1, y1))
                        if y1 > 0:
                            queue.append((x1, y1 - 1))
                        if y1 < width - 1:
                            queue.append((x1, y1 + 1))
                        visited[x1][y1] = 1
                        x.append(x1)
                        y.append(y1)

                if len(x) != 0:
                    objects.append((x, y))
    return objects


def getDimensions(object):
    min_height = object[0][0]
    max_height = object[0][0]
    min_width = object[1][0]
    max_width = object[1][0]

    for i in range(len(object[0])):
        min_height = min(min_height, object[0][i])
        max_height = max(max_height, object[0][i])
        min_width = min(min_width, object[1][i])
        max_width = max(max_width, object[1][i])

    return min_height, max_height, min_width, max_width


def getCenter(object):
    min_height, max_height, min_width, max_width = getDimensions(object)
    return int((max_height + min_height) / 2), int((max_width + min_width) / 2)


def getFill(object):
    min_height, max_height, min_width, max_width = getDimensions(object)
    return len(object)/((max_height - min_height + 1) * (max_width - min_width + 1))


def showObjects(bitMap, objects):
    plt.imshow(bitMap)
    for object in objects:
        min_height, max_height, min_width, max_width = getDimensions(object)
        plt.plot((min_width, max_width, max_width, min_width, min_width),
                 (min_height, min_height, max_height, max_height, min_height),
                 color='r')
    plt.show()


def filterObjects(objects, bitMap):
    # Size filters
    min_height_threshold = size_x
    max_height_threshold = 150
    min_width_threshold = size_y
    max_width_threshold = 150

    # Fill filters
    min_filter = 0
    max_filter = .5

    filteredObjects = []

    for i in range(len(objects)):
        progress_bar(i + 1, len(objects))

        object = objects[i]
        accepted = 1
        min_height, max_height, min_width, max_width = getDimensions(object)

        if max_height - min_height + 1 < min_height_threshold:
            accepted = 0
        if max_height - min_height + 1 > max_height_threshold:
            accepted = 0
        if max_width - min_width + 1 < min_width_threshold:
            accepted = 0
        if max_width - min_width + 1 > max_width_threshold:
            accepted = 0

        fill = getFill(object)
        if fill < min_filter:
            accepted = 0
        if fill > max_filter:
            accepted = 0

        if accepted:
            # Filter objects classified as noise
            objectBitMap = getObjectBitMap(object, bitMap)
            objectBitMap = resize(objectBitMap, size_x, size_y)
            objectBitMap = torch.unsqueeze(torch.unsqueeze(objectBitMap, 0), 0)
            letter = torch.sigmoid(net(objectBitMap))
            if torch.max(letter) < 0.7:
                accepted = 0

        if accepted == 1:
            filteredObjects.append(object)
        else:
            for i in range(len(object[0])):
                bitMap[object[0][i]][object[1][i]] = 1
    return filteredObjects


def getObjectBitMap(object, bitMap):
    min_height, max_height, min_width, max_width = getDimensions(object)
    objectBitMap = bitMap[min_height:max_height:1, min_width:max_width:1]
    return objectBitMap


def resize(bitMap, height, width):
    newBitMap = torch.zeros((height, width))
    for i in range(height):
        for j in range(width):
            newBitMap[i][j] = bitMap[(int)(i * len(bitMap) / height)][(int)(j * len(bitMap[0]) / width)]
    return newBitMap


# Returns True if letter1 must go after letter2
def after(letter1, letter2, cell):
    if int(letter1[1][0] / cell[0]) > int(letter2[1][0] / cell[0]):
        return True
    elif int(letter1[1][0] / cell[0]) == int(letter2[1][0] / cell[0]):
        if letter1[1][1] > letter2[1][1]:
            return True
        elif letter1[1][1] == letter2[1][1]:
            if letter2[1][0] > letter2[1][0]:
                return True
    return False


# Sort letters by the order of appearance
def arrange(letters, cell):
    for i in range(len(letters)):
        for j in range(i + 1, len(letters)):
            if after(letters[i], letters[j], cell):
                letters[i], letters[j] = letters[j], letters[i]


# Returns a 2d-list of words arranged in order, if correction is set to True words are replaced with regular expressions
def getOutput(letters, cell, correction=False):
    uncertainty_threshhold = 0.96

    output = []
    line = []
    output_stats = []
    line_stats = []
    word_stats = [letters[0][3]]

    word = ""
    # if not correction or letters[0][2] > 0.96:
    #     word = letters[0][0]
    # else:
    #     word = "."

    for i in range(0, len(letters)):
        center_i = letters[i][1]
        center_prev = letters[i - 1][1]

        if int(center_i[0] / cell[0]) > int(center_prev[0] / cell[0]):
            if correction:
                word = '^' + word + '$'
            line.append(word)
            line_stats.append(word_stats)
            word = ""
            word_stats = []
            output.append(line)
            output_stats.append(line_stats)
            line = []
            line_stats = []
        elif int((center_i[1] - center_prev[1]) / cell[1]) > 1:
            if len(word) != 0:
                if correction:
                    word = '^' + word + '$'
                line.append(word)
                line_stats.append(word_stats)
                word = ""
                word_stats = []

        if not correction or letters[i][2] > uncertainty_threshhold:
            word += letters[i][0]
        else:
            word += '.'
        word_stats.append(letters[i][3])

    if correction:
        word = '^' + word + '$'
    line.append(word)
    line_stats.append(word_stats)

    output.append(line)
    output_stats.append(line_stats)

    return output, output_stats


def print_output(output):
    for line in output:
        for word in line:
            print(f"{word} ", end="")
        print()


#########################
# Processing input images
#########################


inputs = 2
input_threshold = 128
correction = True

for input_i in range(inputs):
    print(f"Image {input_i + 1}:\n")
    img = plt.imread(getPath(input_i + 1))

    height = img.shape[0]
    width = img.shape[1]
    bitMap = torch.zeros((height, width))

    # Converting input image to a BitMap
    for i in range(height):
        for j in range(width):
            val = np.mean(img[i][j])
            if val > input_threshold:
                bitMap[i][j] = 1

    print("Detecting Objects")
    objects = getObjects(bitMap, height, width)
    print("\n\nFiltering Objects")
    objects = filterObjects(objects, bitMap)
    showObjects(bitMap, objects)

    print("\nArranging Letters\n")
    letter_sizes = []
    for object in objects:
        min_height, max_height, min_width, max_width = getDimensions(object)
        letter_sizes.append((max_height - min_height, max_width - min_width))
    cell = (int(1.5 * np.mean(letter_sizes[:][0])), int(1.5 * np.mean(letter_sizes[:][1])))

    letters = []
    for object in objects:
        objectBitMap = getObjectBitMap(object, bitMap)
        objectBitMap = resize(objectBitMap, size_x, size_y)

        objectBitMap = torch.unsqueeze(torch.unsqueeze(objectBitMap, 0), 0)
        letter = torch.sigmoid(net(objectBitMap))
        temp = letter
        model_certainty = letter[0][torch.argmax(letter, axis=1)].item()

        idx = torch.argmax(letter, axis=1)
        if idx != len(folder):
            letter = list(folder.class_to_idx.keys())[list(folder.class_to_idx.values()).index(idx)]
            center = getCenter(object)
            letters.append((letter.lower(), center, model_certainty, temp))

    arrange(letters, cell)
    output, output_stats = getOutput(letters, cell, correction)

    # Running regular expressions through a list of English words, replacing them accordingly
    if correction:
        with open('Data/Words/words.txt') as words:
            words = words.read().split('\n')
            p = re.compile('^[a-z]*$')
            words = [x for x in words if p.match(x)]

            new_output = []
            for line_i in range(len(output)):
                line = output[line_i]
                line_stats = output_stats[line_i]
                new_line = []
                for word_i in range(len(line)):
                    word = line[word_i]
                    word_stats = line_stats[word_i]
                    p = re.compile(word)
                    match = [x for x in words if p.match(x)]
                    if len(match) == 1:
                        new_line.append(match[0])
                    elif len(match) > 1:
                        scores = np.ones(len(match))
                        for i in range(len(match)):
                            for j in range(len(match[i])):
                                scores[i] *= word_stats[j][0][ord(match[i][j]) - ord('a')]
                        new_line.append(match[np.argmax(scores)])
                new_output.append(new_line)
            output = new_output

    print("Result:")
    print_output(output)
