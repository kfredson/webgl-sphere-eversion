f = open('./positions.js')
pos = eval(f.read())
f.close()
g = open('./faces.js')
fcs = eval(g.read())
g.close()

nfcs = []
bad_triangles = set()

for x in fcs:
    if 1737 in x or 1735 in x or 1523 in x:
        nfcs.append(x)
        bad_triangles.add(x[0])
        bad_triangles.add(x[1])
        bad_triangles.add(x[2])
npos = []
for x in pos:
    np = dict()
    for y in bad_triangles:
        np[y] = x[y]
    npos.append(np)

temp_str = 'var coordDicts = $pos1; var faces = $faces;'
temp_str = temp_str.replace('$faces',str(nfcs))
temp_str = temp_str.replace('$pos1',str(npos))

g = open('./dataList.js','w')
g.write(temp_str)
g.close()






