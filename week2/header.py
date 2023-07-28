

if __name__ == '__main__':


    everything = []
    cond = False
    with open('output5.txt', 'r', encoding='utf-8') as file:
        curr = []
        for line in file:
            line = line.strip()
            if (line == 'Header: Ana Başarılı Senaryo'):
                curr.append(line + '\n')
                cond = True
            if cond:
                if line == '------------------------------------------------':
                    everything.append(curr)
                    curr = []
                    cond = False
                else:
                    curr.append(line)
    with open('new2.txt', 'w') as out:
        for x in everything:
            for a in x:
                out.write(a)
