all = []
if __name__ == '__main__':
    with open('new2.txt', 'r') as file:
        for line in file:
            all.append(line.replace('Data: ', 'Scenario: '))
            all.append(line.replace('Scenario: ', 'Gherkin: '))

    with open('final5.txt', 'w') as file:
        for a in all:
            file.write(a)