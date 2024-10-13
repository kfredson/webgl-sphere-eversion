import numpy
import math
import random
from numpy.polynomial import Polynomial as P
import torch

#import matplotlib.pyplot as plt
#import matplotlib.tri as mtri
#import matplotlib

def barycentric(faces, pos_list):
    edgeD = dict()
    cIdx = max([x for x in pos_list[0]])+1
    nFaces = []
    for x in faces:
        if frozenset([x[0],x[1]]) not in edgeD:
            edgeD[frozenset([x[0],x[1]])] = cIdx
            for y in pos_list:
                y[cIdx] = 0.5*(y[x[0]]+y[x[1]])
            cIdx += 1
        if frozenset([x[1],x[2]]) not in edgeD:
            edgeD[frozenset([x[1],x[2]])] = cIdx
            for y in pos_list:
                y[cIdx] = 0.5*(y[x[1]]+y[x[2]])
            cIdx += 1
        if frozenset([x[0],x[2]]) not in edgeD:
            edgeD[frozenset([x[0],x[2]])] = cIdx
            for y in pos_list:
                y[cIdx] = 0.5*(y[x[0]]+y[x[2]])
            cIdx += 1
        nFaces.append([x[0],edgeD[frozenset([x[0],x[1]])],edgeD[frozenset([x[0],x[2]])]])
        nFaces.append([x[1],edgeD[frozenset([x[1],x[2]])],edgeD[frozenset([x[0],x[1]])]])
        nFaces.append([x[2],edgeD[frozenset([x[0],x[2]])],edgeD[frozenset([x[1],x[2]])]])
        nFaces.append([edgeD[frozenset([x[0],x[1]])],edgeD[frozenset([x[0],x[2]])],edgeD[frozenset([x[1],x[2]])]])
    return nFaces

def mutate(faces, pos_list, topNode, joints, frac, edge_to_face, interp_type):
    nFaces = []
    nPos = []
    even_joints = set()
    startJoint = None
    for x in joints:
        if topNode in x and topNode in edge_to_face[frozenset(x)]:
            startJoint = x
    evenJoints = set()
    cJoint = startJoint
    while cJoint != None:
        tJoint = cJoint
        cJoint = None
        for y in joints:
            if topNode in y and topNode in edge_to_face[frozenset(y)]:
                if len(y.intersection(tJoint)) == 2 and frozenset(y) not in evenJoints:
                    evenJoints.add(frozenset(y))
                    tJoint = y
                    cJoint = y
    nvert = dict()
    cIdx = max([x for x in pos_list[0]])+1
    sf = 1
    for y in evenJoints:
        center = edge_to_face[y]
        sides = [x for x in y if x not in center]
        opp = [c for c in center if c != topNode]
        print(opp,center,y)
        for x in sides:
            if x in nvert:
                pass
            else:
                for pos in pos_list:
                    #nvert[x] = ((1-frac)*pos[topNode]+frac*pos[x],cIdx)
                    scaleFact = 1
                    rScaleFact = 1
                    nVec = (1-frac)*pos[topNode]+frac*pos[x]
                    if pos != topNode:
                        scaleFact = numpy.linalg.norm(pos[x][0:2])/math.sqrt(3*3-pos[x][2]*pos[x][2])
                        sf = scaleFact
                        rScaleFact = scaleFact*math.sqrt(3*3-nVec[2]*nVec[2])/numpy.linalg.norm(nVec[0:2])
                    nVec = numpy.array([rScaleFact*nVec[0],rScaleFact*nVec[1],nVec[2]])
                    nvert[x] = (nVec,cIdx)
                    pos[cIdx] = nvert[x][0]
                interp_type[cIdx] = interp_type[topNode]
                cIdx += 1
        nFaces.append([sides[1],opp[0],nvert[sides[1]][1]])
        nFaces.append([nvert[sides[0]][1],opp[0],nvert[sides[1]][1]])
        nFaces.append([nvert[sides[0]][1],opp[0],sides[0]])
        nFaces.append([topNode,nvert[sides[0]][1],nvert[sides[1]][1]])
    for x in faces:
        if topNode not in x:
            nFaces.append(x)
    return nFaces

def getMiddle(numLoops,insideProfile,outsideProfile,cintProfile,coutProfile,startIdx,topZ,bottomZ,r,gap):
    pts = dict()
    faces = []
    counter = 0
    ptTrip = []
    bottom_perim = []
    top_perim = []
    #Create first loop
    intR = insideProfile[0]
    outR = outsideProfile[0]
    cintR = cintProfile[0]
    coutR = coutProfile[0]
    outAdjust1 = []
    outAdjust2 = []
    inAdjust1 = []
    inAdjust2 = []
    leveldict = {}
    pairs = []
    pts[counter] = numpy.array([0,0,topZ+gap])
    counter += 1
    pts[counter] = numpy.array([0,0,bottomZ-gap])
    counter += 1
    leveldict[0] = len(insideProfile)-1
    leveldict[1] = 0
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([r*math.cos((2*y+0.75)*math.pi/numLoops),r*math.sin((2*y+0.75)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.0)*math.pi/numLoops),r*math.sin((2*y+1.0)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.25)*math.pi/numLoops),r*math.sin((2*y+1.25)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.75)*math.pi/numLoops),r*math.sin((2*y+1.75)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.0)*math.pi/numLoops),r*math.sin((2*y+2.0)*math.pi/numLoops),bottomZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.25)*math.pi/numLoops),r*math.sin((2*y+2.25)*math.pi/numLoops),bottomZ])
        counter += 1
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops),1])
        for i in range(-6,0):
            leveldict[counter+i] = 0
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+0.4)*math.pi/numLoops),outR[0]*math.sin((2*y+0.4)*math.pi/numLoops),outR[1]])
        counter += 1
        pts[counter] = numpy.array([coutR[0]*math.cos((2*y+1.0)*math.pi/numLoops),coutR[0]*math.sin((2*y+1.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+1.6)*math.pi/numLoops),outR[0]*math.sin((2*y+1.6)*math.pi/numLoops),outR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+1.4)*math.pi/numLoops),intR[0]*math.sin((2*y+1.4)*math.pi/numLoops),intR[1]])
        counter += 1
        pts[counter] = numpy.array([cintR[0]*math.cos((2*y+2.0)*math.pi/numLoops),cintR[0]*math.sin((2*y+2.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+2.6)*math.pi/numLoops),intR[0]*math.sin((2*y+2.6)*math.pi/numLoops),intR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*y+x-6*numLoops])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*y+x+1)%(6*numLoops)])
        for i in range(-6,0):
            leveldict[counter+i] = 0 
    for y in range(1,len(insideProfile)-1):
        baseCounter = counter
        for z in range(numLoops):
            outR = outsideProfile[y]
            intR = insideProfile[y]
            cintR = cintProfile[y]
            coutR = coutProfile[y]
            pts[counter] = numpy.array([outR[0]*math.cos((2*z+0.4)*math.pi/numLoops),outR[0]*math.sin((2*z+0.4)*math.pi/numLoops),outR[1]])
            counter += 1
            pts[counter] = numpy.array([coutR[0]*math.cos((2*z+1.0)*math.pi/numLoops),coutR[0]*math.sin((2*z+1.0)*math.pi/numLoops),coutR[1]])
            counter += 1
            pts[counter] = numpy.array([outR[0]*math.cos((2*z+1.6)*math.pi/numLoops),outR[0]*math.sin((2*z+1.6)*math.pi/numLoops),outR[1]])
            counter += 1
            pairs.append([counter-1,counter-2,counter-3])
            pts[counter] = numpy.array([intR[0]*math.cos((2*z+1.4)*math.pi/numLoops),intR[0]*math.sin((2*z+1.4)*math.pi/numLoops),intR[1]])
            counter += 1
            pts[counter] = numpy.array([cintR[0]*math.cos((2*z+2.0)*math.pi/numLoops),cintR[0]*math.sin((2*z+2.0)*math.pi/numLoops),cintR[1]])
            counter += 1
            pts[counter] = numpy.array([intR[0]*math.cos((2*z+2.6)*math.pi/numLoops),intR[0]*math.sin((2*z+2.6)*math.pi/numLoops),intR[1]])
            counter += 1
            pairs.append([counter-1,counter-2,counter-3])
            for x in range(6):
                faces.append([baseCounter+6*z+x,baseCounter+(6*z+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*z+x-6*numLoops])
                faces.append([baseCounter+6*z+x,baseCounter+(6*z+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*z+x+1)%(6*numLoops)])
            for i in range(-6,0):
                leveldict[counter+i] = y 
    intR = insideProfile[-1]
    outR = outsideProfile[-1]
    cintR = cintProfile[-1]
    coutR = coutProfile[-1]
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+0.4)*math.pi/numLoops),outR[0]*math.sin((2*y+0.4)*math.pi/numLoops),outR[1]])
        counter += 1
        pts[counter] = numpy.array([coutR[0]*math.cos((2*y+1.0)*math.pi/numLoops),coutR[0]*math.sin((2*y+1.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([outR[0]*math.cos((2*y+1.6)*math.pi/numLoops),outR[0]*math.sin((2*y+1.6)*math.pi/numLoops),outR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+1.4)*math.pi/numLoops),intR[0]*math.sin((2*y+1.4)*math.pi/numLoops),intR[1]])
        counter += 1
        pts[counter] = numpy.array([cintR[0]*math.cos((2*y+2.0)*math.pi/numLoops),cintR[0]*math.sin((2*y+2.0)*math.pi/numLoops),coutR[1]])
        counter += 1
        pts[counter] = numpy.array([intR[0]*math.cos((2*y+2.6)*math.pi/numLoops),intR[0]*math.sin((2*y+2.6)*math.pi/numLoops),intR[1]])
        counter += 1
        pairs.append([counter-1,counter-2,counter-3])
        for i in range(-6,0):
                leveldict[counter+i] = len(insideProfile)-1
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*y+x-6*numLoops])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*y+x+1)%(6*numLoops)])
    baseCounter = counter
    for y in range(numLoops):
        pts[counter] = numpy.array([r*math.cos((2*y+0.75)*math.pi/numLoops),r*math.sin((2*y+0.75)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.0)*math.pi/numLoops),r*math.sin((2*y+1.0)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.25)*math.pi/numLoops),r*math.sin((2*y+1.25)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+1.75)*math.pi/numLoops),r*math.sin((2*y+1.75)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.0)*math.pi/numLoops),r*math.sin((2*y+2.0)*math.pi/numLoops),topZ])
        counter += 1
        pts[counter] = numpy.array([r*math.cos((2*y+2.25)*math.pi/numLoops),r*math.sin((2*y+2.25)*math.pi/numLoops),topZ])
        counter += 1
        for i in range(-6,0):
                leveldict[counter+i] = len(insideProfile)-1
        for x in range(6):
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+6*y+x-6*numLoops])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops)-6*numLoops,baseCounter+(6*y+x+1)%(6*numLoops)])
                faces.append([baseCounter+6*y+x,baseCounter+(6*y+x+1)%(6*numLoops),0])
    return (faces,bottom_perim,top_perim,pts,leveldict,pairs)

def det(a,b,c,d,e,f,g,h,i):
    return a*e*i-a*f*h-(b*d*i-b*g*f)+(c*d*h-c*e*g)

def createFaceToEdge(faces):
    edgeD = dict()
    for x in faces:
        if frozenset([x[0],x[1]]) in edgeD:
            edgeD[frozenset([x[0],x[1]])].append(x)
        else:
            edgeD[frozenset([x[0],x[1]])] = [x]
        if frozenset([x[1],x[2]]) in edgeD:
            edgeD[frozenset([x[1],x[2]])].append(x)
        else:
            edgeD[frozenset([x[1],x[2]])] = [x]
        if frozenset([x[0],x[2]]) in edgeD:
            edgeD[frozenset([x[0],x[2]])].append(x)
        else:
            edgeD[frozenset([x[0],x[2]])] = [x]
    return edgeD

def createJoints(faceToEdge):
    joints = []
    for y in faceToEdge:
        x = faceToEdge[y]
        joints.append((set(x[0])).union(set(x[1])))
    return joints

def createEdgeToFace(faceToEdge):
    edge_to_face = dict()
    for y in faceToEdge:
        x = faceToEdge[y]
        edge_to_face[frozenset(set(x[0]).union(set(x[1])))] = (set(x[0])).intersection(set(x[1]))
    return edge_to_face

def createTriangleToVertex(faces):
    triangle_to_vertex = dict()
    for x in faces:
        for y in faces:
            x0 = set(x)
            y0 = set(y)
            if len(x0.intersection(y0))==1:
                intV = list(x0.intersection(y0))[0]
                if (frozenset(x),intV) not in triangle_to_vertex:
                    triangle_to_vertex[(frozenset(x),intV)] = set()
                for q in y:
                    if q != intV:
                        triangle_to_vertex[(frozenset(x),intV)].add(q)
    return triangle_to_vertex

def vertexToEdge(faces):
    vDict = dict()
    for x in faces:
        for y in x:
            if y not in vDict:
                vDict[y] = set()
            for z in x:
                if z != y:
                    vDict[y].add(z)
    return vDict
            

def secondOrderEdge(vertex_to_edge):
    sDict = dict()
    for x in vertex_to_edge:
        if x not in sDict:
            sDict[x] = set()
        for q in vertex_to_edge[x]:
            for y in vertex_to_edge[q]:
                if y!=x:
                    sDict[x].add(y)
    return sDict

def getRepelIndices(second_order_edge):
    index1 = []
    index2 = []
    for x in second_order_edge:
        for y in second_order_edge[x]:
            index1.append(x)
            index2.append(y)
    return [torch.tensor(index1),torch.tensor(index2)]

def createInverseDict(faces,pos):
    inverseDict = dict()
    for x in faces:
        inverseDict[frozenset(x)] = numpy.linalg.inv([pos[x[0]],pos[x[1]],pos[x[2]]])
    return inverseDict

def getMinDist(coords,edge_coords):
    d1 = numpy.linalg.norm(coords[0]-edge_coords[1])
    d2 = numpy.linalg.norm(coords[1]-edge_coords[1])
    d3 = numpy.linalg.norm(coords[0]-edge_coords[0])
    d4 = numpy.linalg.norm(coords[0]-edge_coords[1])
    d5 = numpy.linalg.norm(edge_coords[0]-edge_coords[1])
    return min([d1,d2,d3,d4,d5])

LENGTH_THRESH = 0.001
ANGLE_THRESH = 0.99

def getAngle(coords,edge_coords):
    n1 = numpy.cross(coords[0]-edge_coords[1],edge_coords[0]-edge_coords[1])
    n1 = n1/numpy.linalg.norm(n1)    
    n2 = numpy.cross(coords[1]-edge_coords[1],edge_coords[0]-edge_coords[1])
    n2 = n2/numpy.linalg.norm(n2)
    return numpy.dot(n1,n2)

def isSingular(pt1,pt2,edge_to_face,joints,fList):
    linear1 = dict()
    for x in pt1:
        linear1[x] = [P([pt1[x][0],pt2[x][0]-pt1[x][0]]),P([pt1[x][1],pt2[x][1]-pt1[x][1]]),P([pt1[x][2],pt2[x][2]-pt1[x][2]])]
    #detPolys = dict()
    for x in joints:
        combo = sorted([y for y in x])
        mat1 = [[(pt2[combo[1]][0]-pt1[combo[1]][0])-(pt2[combo[0]][0]-pt1[combo[0]][0]),
                (pt2[combo[1]][1]-pt1[combo[1]][1])-(pt2[combo[0]][1]-pt1[combo[0]][1]),
                (pt2[combo[1]][2]-pt1[combo[1]][2])-(pt2[combo[0]][2]-pt1[combo[0]][2])],
                [(pt2[combo[2]][0]-pt1[combo[2]][0])-(pt2[combo[0]][0]-pt1[combo[0]][0]),
                (pt2[combo[2]][1]-pt1[combo[2]][1])-(pt2[combo[0]][1]-pt1[combo[0]][1]),
                (pt2[combo[2]][2]-pt1[combo[2]][2])-(pt2[combo[0]][2]-pt1[combo[0]][2])],
                [(pt2[combo[3]][0]-pt1[combo[3]][0])-(pt2[combo[0]][0]-pt1[combo[0]][0]),
                (pt2[combo[3]][1]-pt1[combo[3]][1])-(pt2[combo[0]][1]-pt1[combo[0]][1]),
                (pt2[combo[3]][2]-pt1[combo[3]][2])-(pt2[combo[0]][2]-pt1[combo[0]][2])]]
        mat2 = [[pt1[combo[1]][0]-pt1[combo[0]][0],
                pt1[combo[1]][1]-pt1[combo[0]][1],
                pt1[combo[1]][2]-pt1[combo[0]][2]],
                [pt1[combo[2]][0]-pt1[combo[0]][0],
                pt1[combo[2]][1]-pt1[combo[0]][1],
                pt1[combo[2]][2]-pt1[combo[0]][2]],
                [pt1[combo[3]][0]-pt1[combo[0]][0],
                pt1[combo[3]][1]-pt1[combo[0]][1],
                pt1[combo[3]][2]-pt1[combo[0]][2]]]
        c3 = det(*(mat1[0]+mat1[1]+mat1[2]))
        c2 = det(*(mat1[0]+mat1[1]+mat2[2]))+det(*(mat1[0]+mat2[1]+mat1[2]))+det(*(mat2[0]+mat1[1]+mat1[2]))
        c1 = det(*(mat2[0]+mat2[1]+mat1[2]))+det(*(mat2[0]+mat1[1]+mat2[2]))+det(*(mat1[0]+mat2[1]+mat2[2]))
        c0 = det(*(mat2[0]+mat2[1]+mat2[2]))
        arr = [100*c3,100*c2,100*c1,100*c0]
        r = numpy.roots(arr)
        realR = [(x,numpy.real(t)) for t in r if numpy.imag(t)==0 and numpy.real(t) >= 0 and numpy.real(t) <= 1]
        if arr[-1]*sum(arr) < 0 and len(realR)==0:
            print(("Problem",x,arr))
        edges1 = edge_to_face[frozenset(x)]
        for y in realR:
            coords = [numpy.array([linear1[j][i](y[1]) for i in range(3)]) for j in y[0] if j not in edges1]
            edge_coords = [numpy.array([linear1[j][i](y[1]) for i in range(3)]) for j in y[0] if j in edges1]
            #print(getAngle(coords,edge_coords))
            if getAngle(coords,edge_coords) >= ANGLE_THRESH or getMinDist(coords,edge_coords) <= LENGTH_THRESH:
                #print((coords,edge_coords,x[0]))
                #print('###########')
                #for z in y[0]:
                #    print(fList[z])
                return True
    return False

def isSingularP(pt1,pt2,edge_to_face,joints,fList,selectedPoint):
    linear1 = dict()
    for x in pt1:
        linear1[x] = [P([pt1[x][0],pt2[x][0]-pt1[x][0]]),P([pt1[x][1],pt2[x][1]-pt1[x][1]]),P([pt1[x][2],pt2[x][2]-pt1[x][2]])]
    #detPolys = dict()
    for x in joints:
        combo = sorted([y for y in x])
        mat1 = [[(pt2[combo[1]][0]-pt1[combo[1]][0])-(pt2[combo[0]][0]-pt1[combo[0]][0]),
                (pt2[combo[1]][1]-pt1[combo[1]][1])-(pt2[combo[0]][1]-pt1[combo[0]][1]),
                (pt2[combo[1]][2]-pt1[combo[1]][2])-(pt2[combo[0]][2]-pt1[combo[0]][2])],
                [(pt2[combo[2]][0]-pt1[combo[2]][0])-(pt2[combo[0]][0]-pt1[combo[0]][0]),
                (pt2[combo[2]][1]-pt1[combo[2]][1])-(pt2[combo[0]][1]-pt1[combo[0]][1]),
                (pt2[combo[2]][2]-pt1[combo[2]][2])-(pt2[combo[0]][2]-pt1[combo[0]][2])],
                [(pt2[combo[3]][0]-pt1[combo[3]][0])-(pt2[combo[0]][0]-pt1[combo[0]][0]),
                (pt2[combo[3]][1]-pt1[combo[3]][1])-(pt2[combo[0]][1]-pt1[combo[0]][1]),
                (pt2[combo[3]][2]-pt1[combo[3]][2])-(pt2[combo[0]][2]-pt1[combo[0]][2])]]
        mat2 = [[pt1[combo[1]][0]-pt1[combo[0]][0],
                pt1[combo[1]][1]-pt1[combo[0]][1],
                pt1[combo[1]][2]-pt1[combo[0]][2]],
                [pt1[combo[2]][0]-pt1[combo[0]][0],
                pt1[combo[2]][1]-pt1[combo[0]][1],
                pt1[combo[2]][2]-pt1[combo[0]][2]],
                [pt1[combo[3]][0]-pt1[combo[0]][0],
                pt1[combo[3]][1]-pt1[combo[0]][1],
                pt1[combo[3]][2]-pt1[combo[0]][2]]]
        c3 = det(*(mat1[0]+mat1[1]+mat1[2]))
        c2 = det(*(mat1[0]+mat1[1]+mat2[2]))+det(*(mat1[0]+mat2[1]+mat1[2]))+det(*(mat2[0]+mat1[1]+mat1[2]))
        c1 = det(*(mat2[0]+mat2[1]+mat1[2]))+det(*(mat2[0]+mat1[1]+mat2[2]))+det(*(mat1[0]+mat2[1]+mat2[2]))
        c0 = det(*(mat2[0]+mat2[1]+mat2[2]))
        arr = [100*c3,100*c2,100*c1,100*c0]
        r = numpy.roots(arr)
        realR = [(x,numpy.real(t)) for t in r if numpy.imag(t)==0 and numpy.real(t) >= 0 and numpy.real(t) <= 1]
        #if arr[-1]*sum(arr) < 0 and len(realR)==0:
        #    print(("Problem",x,arr))
        edges1 = edge_to_face[frozenset(x)]
        for y in realR:
            coords = [numpy.array([linear1[j][i](y[1]) for i in range(3)]) for j in y[0] if j not in edges1]
            edge_coords = [numpy.array([linear1[j][i](y[1]) for i in range(3)]) for j in y[0] if j in edges1]
            #print(getAngle(coords,edge_coords))
            if getAngle(coords,edge_coords) >= ANGLE_THRESH or getMinDist(coords,edge_coords) <= LENGTH_THRESH:
                #print((coords,edge_coords,x[0]))
                print('###########')
                for z in y[0]:
                    print(z)
                return True
    return False

def isSingular2(pt1,pt2,triangle_to_vertex,fList,singTrSet):
    linear1 = dict()
    for x in pt1:
        linear1[x] = [P([pt1[x][0],pt2[x][0]-pt1[x][0]]),P([pt1[x][1],pt2[x][1]-pt1[x][1]]),P([pt1[x][2],pt2[x][2]-pt1[x][2]])]
    
    tList2 = []
    for y in triangle_to_vertex:
        detPolys = dict()
        for x in triangle_to_vertex[y]:
            basept = y[1]
            ft = [z for z in y[0] if z != basept]
            mat1 =  [linear1[ft[0]][0]-linear1[basept][0],linear1[ft[0]][1]-linear1[basept][1],linear1[ft[0]][2]-linear1[basept][2],
                     linear1[ft[1]][0]-linear1[basept][0],linear1[ft[1]][1]-linear1[basept][1],linear1[ft[1]][2]-linear1[basept][2],
                     linear1[x][0]-linear1[basept][0],linear1[x][1]-linear1[basept][1],linear1[x][2]-linear1[basept][2]]
            detPolys[x] = det(*mat1)
        for x in detPolys:
            coefs = list(detPolys[x].coef)
            coefs.reverse()
            r = numpy.roots(coefs)
            realR = [(y,x,numpy.real(t)) for t in r if numpy.imag(t)==0 and t > 0 and t < 1]
            if detPolys[x](0)*detPolys[x](1) < 0 and len(realR)==0:
                print(("Problem",x))
            tList2.extend(realR)

    print(tList2)
    badEdges = []
    for x in tList2:
        basept = x[0][1]
        ft = [z for z in x[0][0] if z != basept]
        tVec1 = numpy.array([linear1[ft[0]][i](x[2])-linear1[basept][i](x[2]) for i in range(3)])
        tVec2 = numpy.array([linear1[ft[1]][i](x[2])-linear1[basept][i](x[2]) for i in range(3)])
        oVec = numpy.array([linear1[x[1]][i](x[2])-linear1[basept][i](x[2]) for i in range(3)])
        tVec1 = tVec1/numpy.linalg.norm(tVec1)
        tVec2 = tVec2/numpy.linalg.norm(tVec2)
        oVec = oVec/numpy.linalg.norm(oVec)
        cos1 = numpy.dot(oVec,tVec1)
        cos2 = numpy.dot(oVec,tVec2)
        c1 = numpy.cross(tVec1,oVec)
        c2 = numpy.cross(oVec,tVec2)
        c3 = numpy.cross(tVec2,tVec1)
        if (numpy.dot(c1,c2) > 0 and numpy.dot(c1,c3) < 0): #or cos1 >= 0.999 or cos2 >= 0.999:
            print('###########')
            print(x[0])
            singTrSet.add(x[0][0])
            #return True
    return False

def interpolate(pos1,pos2,t):
    pos3 = dict()
    for x in pos1:
        pos3[x] = (1-t)*pos1[x]+t*pos2[x]
    return pos3

def getTwistMapping(pos1,pos2,t,interp_type):
    pos3 = dict()
    offset = numpy.array([0,0,0])
    for x in pos1:
        if interp_type[x]==1:
            start = pos1[x]-offset;
            end = pos2[x]-offset;
            midpoint = (start+end)/2.;
            nmidpoint = midpoint/numpy.linalg.norm(midpoint)
            d1 = numpy.dot(start,nmidpoint);
            d2 = numpy.dot(end,nmidpoint);
            pstart = start-d1*nmidpoint;
            pend = end-d2*nmidpoint;
            b2 = numpy.cross(nmidpoint,pstart);

            rot = math.cos(math.pi*t)*pstart+math.sin(math.pi*t)*b2;
            pos3[x] = rot+((1.-t)*d1+t*d2)*nmidpoint+offset;
        elif interp_type[x]==2:
            start = pos1[x];
            end = pos2[x];
            midpoint = (start+end)/2.;
            start = pos1[x]-midpoint;
            end = pos2[x]-midpoint;
            nmidpoint = numpy.array([0,0,-1])
            d1 = numpy.dot(start,nmidpoint);
            d2 = numpy.dot(end,nmidpoint);
            pstart = start-d1*nmidpoint;
            pend = end-d2*nmidpoint;
            b2 = numpy.cross(nmidpoint,pstart);

            rot = math.cos(math.pi*t)*pstart+math.sin(math.pi*t)*b2;
            pos3[x] = rot+((1.-t)*d1+t*d2)*nmidpoint+midpoint;
        elif interp_type[x]==3:
            start = pos1[x]-offset;
            end = pos2[x]-offset;
            midpoint = (start+end)/2.;
            start = pos1[x]-midpoint;
            end = pos2[x]-midpoint;
            nmidpoint = numpy.array([0,0,1])
            d1 = numpy.dot(start,nmidpoint);
            d2 = numpy.dot(end,nmidpoint);
            pstart = start-d1*nmidpoint;
            pend = end-d2*nmidpoint;
            b2 = numpy.cross(nmidpoint,pstart);

            rot = math.cos(math.pi*t)*pstart+math.sin(math.pi*t)*b2;
            pos3[x] = rot+((1.-t)*d1+t*d2)*nmidpoint+midpoint;
    return pos3

def getRefLengths(faces,pos):
    ref_lengths = dict()
    for x in faces:
        y = sorted(x)
        ref_lengths[(y[0],y[1])] = numpy.linalg.norm(pos[y[0]]-pos[y[1]])
        ref_lengths[(y[0],y[2])] = numpy.linalg.norm(pos[y[2]]-pos[y[1]])
        ref_lengths[(y[1],y[2])] = numpy.linalg.norm(pos[y[2]]-pos[y[0]])
    return ref_lengths

def getPtToFace(edge_to_face):
    pt_to_face = dict()
    for x in edge_to_face:
        for y in x:
            if y in pt_to_face:
                pt_to_face[y].append(x)
            else:
                pt_to_face[y] = [x]
    return pt_to_face

def gs(v1, v2):
    if numpy.linalg.norm(v1) < 0.000001:
        print("Warning")
        return numpy.identity(2)
    c11 = 1/numpy.linalg.norm(v1)
    c12 = 0
    rescaledV1 = v1*c11
    dp = numpy.dot(rescaledV1,v2)
    diffV2 = v2-dp*rescaledV1
    diffV2_norm = numpy.linalg.norm(diffV2)
    #if diffV2_norm <= 0.000001:
    #    return numpy.identity(2)
    c21 = -dp*c11/diffV2_norm
    c22 = 1/diffV2_norm
    return numpy.array([[c11,c12],[c21,c22]])

def assembleTensors(ref_positions,edge_to_face):
    fcs = [x for x in edge_to_face]
    fcs.sort()
    a11T = []
    a12T = []
    a21T = []
    a22T = []
    areas = []
    for face in fcs:
        edge = edge_to_face[face]
        edge = [z for z in edge]
        edge.sort()
        opp_pts = [z for z in face if z not in edge]
        opp_pts.sort()
        for e in edge:
            v0base = ref_positions[opp_pts[0]]-ref_positions[e]
            v1base = ref_positions[opp_pts[1]]-ref_positions[e]
            #v0 = pos2[opp_pts[0]]-pos2[e]
            #v1 = pos2[opp_pts[1]]-pos2[e]
            m1 = gs(v0base,v1base)
            #A = numpy.array([v0,v1])
            B = numpy.transpose(m1)
            a11 = 0.25*B[0][0]
            a12 = 0.25*B[0][1]
            a21 = 0.25*B[1][0]
            a22 = 0.25*B[1][1]
            a11T.append(a11)
            a12T.append(a12)
            a21T.append(a21)
            a22T.append(a22)
            areas.append(numpy.linalg.norm(numpy.cross(v0base,v1base)))
        '''for opp_pt in opp_pts:
            v0base = ref_positions[edge[0]]-ref_positions[opp_pt]
            v1base = ref_positions[edge[1]]-ref_positions[opp_pt]
            #v0 = pos2[edge[0]]-pos2[opp_pt]
            #v1 = pos2[edge[1]]-pos2[opp_pt]
            m1 = gs(v0base,v1base)
            #A = numpy.array([v0,v1])
            B = numpy.transpose(m1)
            a11 = B[0][0]
            a12 = B[0][1]
            a21 = B[1][0]
            a22 = B[1][1]
            a11T.append(a11)
            a12T.append(a12)
            a21T.append(a21)
            a22T.append(a22)
            areas.append(numpy.linalg.norm(numpy.cross(v0base,v1base)))'''
    return [a11T,a12T,a21T,a22T,areas]

def getIndices(edge_to_face):
    fcs = [x for x in edge_to_face]
    fcs.sort()
    aIndex = []
    bIndex = []
    cIndex = []
    for face in fcs:
        edge = edge_to_face[face]
        edge = [z for z in edge]
        edge.sort()
        opp_pts = [z for z in face if z not in edge]
        opp_pts.sort()
        for e in edge:
            aIndex.append(opp_pts[0])
            bIndex.append(opp_pts[1])
            cIndex.append(e)
        '''for opp_pt in opp_pts:
            aIndex.append(edge[0])
            bIndex.append(edge[1])
            cIndex.append(opp_pt)'''
    return [aIndex,bIndex,cIndex]

def relaxElasticEnergyTensor(pos2,indices,derivs,cGrad):
    cGrad = [cGrad[0]*0.0+derivs[0],cGrad[1]*0.0+derivs[1],cGrad[2]*0.0+derivs[2]]
    pos2.index_add_(0,indices[0],cGrad[0],alpha=-0.00002)
    pos2.index_add_(0,indices[1],cGrad[1],alpha=-0.00002)
    pos2.index_add_(0,indices[2],cGrad[2],alpha=-0.00002)
    return cGrad

def relaxTensor(localGrads,cArr,cGrads,sp,crossGrads,rp,repel,indices,repelIndices):
    for x in enumerate(crossGrads):
        if x[0]!=0 and x[0]!= len(crossGrads)-1:
            crossGrads[x[0]] = cArr[x[0]-1]+cArr[x[0]+1]-2*cArr[x[0]]
            cArr[x[0]] = cArr[x[0]]+sp*crossGrads[x[0]]
    for x in enumerate(localGrads):
        if x[0]!=0 and x[0]!= len(localGrads)-1:
            cGrads[x[0]] = [cGrads[x[0]][0]*0.0+x[1][0],
                            cGrads[x[0]][1]*0.0+x[1][1],
                            cGrads[x[0]][2]*0.0+x[1][2]]
            cArr[x[0]].index_add_(0,indices[0],cGrads[x[0]][0],alpha=-0.00000001)
            cArr[x[0]].index_add_(0,indices[1],cGrads[x[0]][1],alpha=-0.00000001)
            cArr[x[0]].index_add_(0,indices[2],cGrads[x[0]][2],alpha=-0.00000001)
    #for x in enumerate(repel):
    #    if x[0]!=0 and x[0]!= len(localGrads)-1:
    #        cArr[x[0]].index_add_(0,repelIndices[0],repel[x[0]],alpha=rp)
    return (cGrads,crossGrads)

'''def harmonic(cArr,vertex_to_edge,factor):
    ncArr = []
    ncArr.append(cArr[0])
    for y in range(len(cArr)):
        if y!=0 and y!= len(cArr)-1:
            narr = []
            for z in range(len(cArr[y])):
                divisor = (1-factor)/len(vertex_to_edge[z])
                npos = factor*cArr[y][z]
                if z==0:
                    print(z)
                    print(cArr[y][z])
                for u in vertex_to_edge[z]:
                    npos += divisor*cArr[y][u]
                    if z==0:
                        print(cArr[y][u])
                if z==0:
                    print(npos)
                    print('$$$$$$$$$$')
                narr.append(npos)
            ncArr.append(narr)
    ncArr.append(cArr[-1])
    return ncArr'''


def harmonic(cArr,vertex_to_edge,factor,blacklist):
    #ncArr = []
    #ncArr.append(cArr[0])
    for y in range(len(cArr)):
        if y!=0 and y!= len(cArr)-1:
            narr = []
            for z in range(len(cArr[y])):
                divisor = (1-factor)/len(vertex_to_edge[z])
                npos = factor*cArr[y][z]
                #if z==0:
                #    print(z)
                #    print(cArr[y][z])
                for u in vertex_to_edge[z]:
                    npos += divisor*cArr[y][u]
                    #if z==0:
                    #    print(cArr[y][u])
                #if z==0:
                #    print(npos)
                #    print('$$$$$$$$$$')
                #narr.append(npos)
                if z not in blacklist:
                    cArr[y][z][0] = npos[0]
                    cArr[y][z][1] = npos[1]
                    cArr[y][z][2] = npos[2]
            #ncArr.append(narr)
    #ncArr.append(cArr[-1])
    #return ncArr


    
                    

def derivA1A1(a1,a2,a3,b1,b2,b3,c1,c2,c3,var):
    if var=='a1':
        return 2*(a1-c1)
    elif var=='a2':
        return 2*(a2-c2)
    elif var=='a3':
        return 2*(a3-c3)
    elif var=='c1':
        return -2*(a1-c1)
    elif var=='c2':
        return -2*(a2-c2)
    elif var=='c3':
        return -2*(a3-c3)
    else:
        return 0

def derivA1A2(a1,a2,a3,b1,b2,b3,c1,c2,c3,var):
    if var=='a1':
        return (b1-c1)
    elif var=='a2':
        return (b2-c2)
    elif var=='a3':
        return (b3-c3)
    elif var=='b1':
        return (a1-c1)
    elif var=='b2':
        return (a2-c2)
    elif var=='b3':
        return (a3-c3)
    elif var=='c1':
        return -(a1-c1)-(b1-c1)
    elif var=='c2':
        return -(a2-c2)-(b2-c2)
    elif var=='c3':
        return -(a3-c3)-(b3-c3)

def derivA2A2(a1,a2,a3,b1,b2,b3,c1,c2,c3,var):
    if var=='b1':
        return 2*(b1-c1)
    elif var=='b2':
        return 2*(b2-c2)
    elif var=='b3':
        return 2*(b3-c3)
    elif var=='c1':
        return -2*(b1-c1)
    elif var=='c2':
        return -2*(b2-c2)
    elif var=='c3':
        return -2*(b3-c3)
    else:
        return 0

def tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2):
    return b1b1*((a1-c1)*(a1-c1)+(a2-c2)*(a2-c2)+(a3-c3)*(a3-c3))+2*b1b2*((a1-c1)*(b1-c1)+(a2-c2)*(b2-c2)+(a3-c3)*(b3-c3))+b2b2*((b1-c1)*(b1-c1)+(b2-c2)*(b2-c2)+(b3-c3)*(b3-c3))


#def det(a1,a2,a3,b1,b2,b3,c1,c2,c3,detb1b2):
#    return detb1b2*(((a1-c1)*(a1-c1)+(a2-c2)*(a2-c2)+(a3-c3)*(a3-c3))*((b1-c1)*(b1-c1)+(b2-c2)*(b2-c2)+(b3-c3)*(b3-c3))-((a1-c1)*(b1-c1)+(a2-c2)*(b2-c2)+(a3-c3)*(b3-c3))*((a1-c1)*(b1-c1)+(a2-c2)*(b2-c2)+(a3-c3)*(b3-c3)))

def derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,var):
    return b1b1*derivA1A1(a1,a2,a3,b1,b2,b3,c1,c2,c3,var)+2*detb1b2*derivA1A2(a1,a2,a3,b1,b2,b3,c1,c2,c3,var)+b2b2*derivA2A2(a1,a2,a3,b1,b2,b3,c1,c2,c3,var)

def derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,var):
    return (b1b1*b2b2-b1b2*b1b2)*(derivA1A1(a1,a2,a3,b1,b2,b3,c1,c2,c3,var)*((b1-c1)*(b1-c1)+(b2-c2)*(b2-c2)+(b3-c3)*(b3-c3))+derivA2A2(a1,a2,a3,b1,b2,b3,c1,c2,c3,var)*((a1-c1)*(a1-c1)+(a2-c2)*(a2-c2)+(a3-c3)*(a3-c3))-2*derivA1A2(a1,a2,a3,b1,b2,b3,c1,c2,c3,var)*((a1-c1)*(b1-c1)+(a2-c2)*(b2-c2)+(a3-c3)*(b3-c3)))


def derivativeA1(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a1')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a1')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a1')
    return areas*rval
    #return rval

def derivativeA2(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a2')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a2')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a2')
    return areas*rval
    #return rval

def derivativeA3(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a3')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a3')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'a3')
    return areas*rval
    #return rval


def derivativeB1(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b1')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b1')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b1')
    return areas*rval
    #return rval


def derivativeB2(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b2')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b2')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b2')
    return areas*rval
    #return rval


def derivativeB3(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b3')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b3')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'b3')
    return areas*rval
    #return rval


def derivativeC1(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c1')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c1')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c1')
    return areas*rval
    #return rval


def derivativeC2(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c2')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c2')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c2')
    return areas*rval
    #return rval


def derivativeC3(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,areas):
    rval = 2*tr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2)*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c3')-2*derivTr(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c3')-2*derivDet(a1,a2,a3,b1,b2,b3,c1,c2,c3,b1b1,b1b2,b2b2,detb1b2,'c3')
    return areas*rval
    #return rval

def round_floats(o):
    if isinstance(o, float): return round(o, 5)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

def reorder_faces(faces,positions):
    for f in range(len(faces)):
        if numpy.dot(numpy.cross(positions[0][faces[f][2]]-positions[0][faces[f][0]],positions[0][faces[f][1]]-positions[0][faces[f][0]]),positions[0][faces[f][0]]) < 0:
            faces[f] = [faces[f][1],faces[f][0],faces[f][2]]

                                                                                            
if __name__ == '__main__':
    p1 = [[1,-2],[2.5,0.5],[2.7,0],[2.5,-0.5],[1,2]]
    p2 = [[0.5,-2],[2.5,1.5],[4.0,0],[2.5,-1.5],[0.5,2]]
    p3 = [[1,-2],[2.5,0.5],[2.3,0],[2.5,-0.5],[1,2]]
    p4 = [[0.5,-2],[2.5,1.5],[4.5,0],[2.5,-1.5],[0.5,2]]
    #p1 = [[x,y] for x,y in p1]
    #p2 = [[x,y] for x,y in p2]
    #p3 =
    numLoops = 14
    mf = getMiddle(numLoops,p2,p1,p4,p3,0,2.5,-2.5,0.5,0.5)

    pairs = mf[5]
    levd = mf[4]
    oldPos = mf[3]
    interp_type = dict()

    newPos = dict()

    for x in pairs:
        if levd[x[0]]==2:
            newPos[x[0]] = oldPos[x[2]]
            newPos[x[2]] = oldPos[x[0]]
            newPos[x[1]] = oldPos[x[1]]
            interp_type[x[0]] = 1
            interp_type[x[1]] = 1
            interp_type[x[2]] = 1
        elif levd[x[0]]==1 or levd[x[0]]==3:
            newPos[x[0]] = numpy.array([oldPos[x[2]][0],oldPos[x[2]][1],-oldPos[x[2]][2]])
            newPos[x[2]] = numpy.array([oldPos[x[0]][0],oldPos[x[0]][1],-oldPos[x[0]][2]])
            newPos[x[1]] = numpy.array([oldPos[x[1]][0],oldPos[x[1]][1],-oldPos[x[1]][2]])
            interp_type[x[0]] = 1
            interp_type[x[1]] = 1
            interp_type[x[2]] = 1
        elif levd[x[0]]==0 or levd[x[0]]==4:
            if numpy.linalg.norm(oldPos[x[0]][0:2]) > 0.75:
                midpt = (oldPos[x[0]][0:2]+oldPos[x[2]][0:2])/2
                subtr = numpy.array([midpt[0],midpt[1],0])
                newPos[x[0]] = oldPos[x[2]]-0.5*subtr
                newPos[x[2]] = oldPos[x[0]]-0.5*subtr
                newPos[x[1]] = oldPos[x[1]]-0.5*subtr
            else:
                newPos[x[0]] = numpy.array([2*oldPos[x[2]][0],2*oldPos[x[2]][1],oldPos[x[2]][2]])
                newPos[x[2]] = numpy.array([2*oldPos[x[0]][0],2*oldPos[x[0]][1],oldPos[x[0]][2]])
                newPos[x[1]] = numpy.array([2*oldPos[x[1]][0],2*oldPos[x[1]][1],oldPos[x[1]][2]])
            if levd[x[0]] == 0:
                interp_type[x[0]] = 3
                interp_type[x[1]] = 3
                interp_type[x[2]] = 3
            else:
                interp_type[x[0]] = 2
                interp_type[x[1]] = 2
                interp_type[x[2]] = 2

    for x in oldPos:
        if x not in newPos:
            newPos[x] = numpy.array([-oldPos[x][0], -oldPos[x][1], oldPos[x][2]])
            if oldPos[x][2] < 0:
                interp_type[x] = 3
            else:
                interp_type[x] = 2

    final_inverted = dict()
    sinv = math.sin(math.pi/4/numLoops)
    cosv = math.cos(math.pi/4/numLoops)
    for x in pairs:
        avg = -(newPos[x[0]]+newPos[x[2]])/2
        tLength = math.sqrt(3*3-avg[2]*avg[2])
        avg2 = tLength*avg[0:2]/numpy.linalg.norm(avg[0:2])
        nAvg = numpy.array([cosv*avg2[0]+sinv*avg2[1],cosv*avg2[1]-sinv*avg2[0],-avg[2]])
        nAvg2 = numpy.array([cosv*avg2[0]-sinv*avg2[1],cosv*avg2[1]+sinv*avg2[0],-avg[2]])
        final_inverted[x[2]] = nAvg
        final_inverted[x[0]] = nAvg2
        final_inverted[x[1]] = numpy.array([avg2[0],avg2[1],-avg[2]])
    for x in newPos:
        if x not in final_inverted and x!=0 and x!=1:
            tLength = math.sqrt(3*3-newPos[x][2]*newPos[x][2])
            avg2 = tLength*newPos[x][0:2]/numpy.linalg.norm(newPos[x][0:2])
            final_inverted[x] = numpy.array([avg2[0],avg2[1],newPos[x][2]])
        elif x==0 or x==1:
            #final_inverted[x] = newPos[x]
            if x==0:
                final_inverted[x] = numpy.array([0.,0.,3.0])
            else:
                final_inverted[x] = numpy.array([0.,0.,-3.0])

    initial = dict()
    for x in final_inverted:
        initial[x] = -1*final_inverted[x]

    face_to_edge = createFaceToEdge(mf[0])
    edge_to_face = createEdgeToFace(face_to_edge)
    joints = createJoints(face_to_edge)
    pos_list = [initial,final_inverted,newPos,oldPos]
    #nf = mutate(mf[0], pos_list, 0, joints, 0.8, edge_to_face, interp_type)
    nf = mf[0]
    face_to_edge = createFaceToEdge(nf)
    edge_to_face = createEdgeToFace(face_to_edge)
    joints = createJoints(face_to_edge)
    for x in range(3):
        nf = mutate(nf, pos_list, 0, joints, 0.8, edge_to_face, interp_type)
        face_to_edge = createFaceToEdge(nf)
        edge_to_face = createEdgeToFace(face_to_edge)
        joints = createJoints(face_to_edge)

    for x in range(4):
        nf = mutate(nf, pos_list, 1, joints, 0.8, edge_to_face, interp_type)
        face_to_edge = createFaceToEdge(nf)
        edge_to_face = createEdgeToFace(face_to_edge)
        joints = createJoints(face_to_edge)

    for x in initial:
        if initial[x][2]==2.5:
            print(x,initial[x])

    '''for pos in pos_list:
                    #nvert[x] = ((1-frac)*pos[topNode]+frac*pos[x],cIdx)
                    scaleFact = 1
                    rScaleFact = 1
                    nVec = (1-frac)*pos[topNode]+frac*pos[x]
                    if pos != topNode:
                        scaleFact = numpy.linalg.norm(pos[x][0:2])/math.sqrt(3*3-pos[x][2]*pos[x][2])
                        rScaleFact = scaleFact*math.sqrt(3*3-nVec[2]*nVec[2])/numpy.linalg.norm(nVec[0:2])
                    nVec = numpy.array([rScaleFact*nVec[0],rScaleFact*nVec[1],nVec[2]])
                    nvert[x] = (nVec,cIdx)
                    pos[cIdx] = nvert[x][0]'''

    for x in final_inverted:
        if x==0 or x==1:
            pass
        else:
            cv = final_inverted[x]
            scaleFact = math.sqrt(3*3-cv[2]*cv[2])/numpy.linalg.norm(cv[0:2])
            nVec = numpy.array([scaleFact*cv[0],scaleFact*cv[1],cv[2]])
            final_inverted[x] = nVec

    '''for x in newPos:
        if x==0 or x==1 or math.abs(newPos[x]) < 2.5:
            pass
        else:

    for x in oldPos:
        if x==0 or x==1 or math.abs(oldPos[x]) < 2.5:
            pass
        else:'''
            

    for x in initial:
        if x==0 or x==1:
            pass
        else:
            cv = initial[x]
            scaleFact = math.sqrt(3*3-cv[2]*cv[2])/numpy.linalg.norm(cv[0:2])
            nVec = numpy.array([scaleFact*cv[0],scaleFact*cv[1],cv[2]])
            initial[x] = nVec
    

    twistArr = [final_inverted]
    for x in range(1,5):
        nx = interpolate(final_inverted,newPos,x/5)
        twistArr.append(nx)
    twistArr.append(newPos)
    for x in range(1,10):
        nx = getTwistMapping(newPos,oldPos,x/10,interp_type)
        twistArr.append(nx)
    twistArr.append(oldPos)
    for x in range(1,5):
        nx = interpolate(oldPos,initial,x/5)
        twistArr.append(nx)
    twistArr.append(initial)
    centerPts = set()

    for x in twistArr:
        z0 = barycentric(nf,[x])
        qf = barycentric(z0,[x])
    nf = qf
    #nf = z0
    face_to_edge = createFaceToEdge(nf)
    edge_to_face = createEdgeToFace(face_to_edge)
    joints = createJoints(face_to_edge)

    for x in range(len(twistArr)):
        for y in twistArr[x]:
            if abs(twistArr[x][y][2]) >= 2.5 and y!=0 and y!=1:
                scaleFact = numpy.linalg.norm(twistArr[x][2][0:2])/math.sqrt(3*3-twistArr[x][2][2]*twistArr[x][2][2])
                rScaleFact = scaleFact*math.sqrt(3*3-twistArr[x][y][2]*twistArr[x][y][2])/numpy.linalg.norm(twistArr[x][y][0:2])
                twistArr[x][y] = numpy.array([rScaleFact*twistArr[x][y][0],rScaleFact*twistArr[x][y][1],twistArr[x][y][2]])
            elif (x==0 or x==len(twistArr)-1) and y!=0 and y!=1:
                scaleFact = numpy.linalg.norm(twistArr[x][2][0:2])/math.sqrt(3*3-twistArr[x][2][2]*twistArr[x][2][2])
                rScaleFact = scaleFact*math.sqrt(3*3-twistArr[x][y][2]*twistArr[x][y][2])/numpy.linalg.norm(twistArr[x][y][0:2])
                twistArr[x][y] = numpy.array([rScaleFact*twistArr[x][y][0],rScaleFact*twistArr[x][y][1],twistArr[x][y][2]])

    for x in range(1,5):
        twistArr[x] = interpolate(twistArr[0],twistArr[5],x/5)

    for x in range(1,5):
        twistArr[15+x] = interpolate(twistArr[15],twistArr[20],x/5)
    

    #for x in range(len(twistArr)-1):
    #    q = isSingular(twistArr[x],twistArr[x+1],edge_to_face,joints,mf[4])
    #    print(x,q)
        #q = isSingular2(twistArr[x],twistArr[x+1],triangle_to_vertex,mf[4])
        #print(x,q)
    #    print('########################################################################################################')

    pt_to_face = getPtToFace(edge_to_face)

    vertices = set()
    for x in edge_to_face:
        for y in x:
            vertices.add(y)
    z = list(vertices)
    print(max(z))
    print(len(z))

    g = getRefLengths(nf,final_inverted)

    '''triangle_to_vertex = createTriangleToVertex(nf)

    #nTwistArr = [twistArr[10]]
    #for y in range(5):
    #    nD = dict()
    #    for x in twistArr[0]:
    #        nD[x] = twistArr[0][x]*5
    #    nTwistArr.append(nD)
    #nTwistArr.append(twistArr[0])
    #twistArr = nTwistArr

    def getLocalGradient(cPos,indices,svecs):
        a1 = torch.index_select(cPos[:,0],0,indices[0])
        a2 = torch.index_select(cPos[:,1],0,indices[0])
        a3 = torch.index_select(cPos[:,2],0,indices[0])
        b1 = torch.index_select(cPos[:,0],0,indices[1])
        b2 = torch.index_select(cPos[:,1],0,indices[1])
        b3 = torch.index_select(cPos[:,2],0,indices[1])
        c1 = torch.index_select(cPos[:,0],0,indices[2])
        c2 = torch.index_select(cPos[:,1],0,indices[2])
        c3 = torch.index_select(cPos[:,2],0,indices[2])
        vecT = [a1,a2,a3,b1,b2,b3,c1,c2,c3]
        vecT.extend(svecs)
        #print(vecT)
        #for y in vecT:
        #    print(y.shape)
        print('derivs')
        derivA1 = torch.stack([derivativeA1(*vecT),
                  derivativeA2(*vecT),
                  derivativeA3(*vecT)])
        derivA1 = torch.transpose(derivA1,0,1)
        #derivA1 = torch.transpose(derivA1[0],0,1)
        derivB1 = torch.stack([derivativeB1(*vecT),
                  derivativeB2(*vecT),
                  derivativeB3(*vecT)])
        derivB1 = torch.transpose(derivB1,0,1)
        #derivB1 = torch.transpose(derivB1[0],0,1)
        derivC1 = torch.stack([derivativeC1(*vecT),
                  derivativeC2(*vecT),
                  derivativeC3(*vecT)])
        derivC1 = torch.transpose(derivC1,0,1)
        return [derivA1,derivB1,derivC1]

    def getRepelGradient(cPos,repelIndices,refLengths):
        c1 = torch.index_select(refLengths,0,repelIndices[0])
        d1 = torch.index_select(refLengths,0,repelIndices[1])
        #repelIndices[0] = point of origin, repelIndices[1] = adjacent points
        a1 = torch.index_select(cPos,0,repelIndices[0])
        b1 = torch.index_select(cPos,0,repelIndices[1])
        ssum = (((b1-a1)*(b1-a1)).sum(dim=1))
        divisor = torch.stack([ssum,ssum,ssum])
        divisor = torch.transpose(divisor,0,1)
        return (a1-b1)/divisor

    cArr = [torch.tensor(numpy.zeros((len(twistArr[0]),3))) for x in range(len(twistArr))]
    for y in range(len(twistArr)):
        for x in twistArr[y]:
            for i in range(3):
                cArr[y][x][i] = twistArr[y][x][i]

    #for x in range(len(cArr)):
    #    cArr[x][1] = cArr[x][1]*(1+(cArr[x][0])**2)/4
    #    cArr[x][2] = cArr[x][2]*(1+(cArr[x][0])**2)/4
        #cArr[x] = 5*ocArr[x]
    output = []
    faceList = [f for f in edge_to_face]

    
    indices = getIndices(edge_to_face)
    indices = [torch.tensor(x) for x in indices]
    svecs = assembleTensors(twistArr[0],edge_to_face)
    svecs = [torch.tensor(x) for x in svecs]
    #b1b1,b1b2,b2b2,detb1b2
    svecs = [svecs[0]*svecs[0]+svecs[1]*svecs[1],svecs[0]*svecs[2]+svecs[1]*svecs[3],svecs[2]*svecs[2]+svecs[3]*svecs[3],svecs[0]*svecs[3]-svecs[1]*svecs[2],svecs[4]]
    cArr = cArr
    cGrad = [torch.tensor(numpy.zeros((len(twistArr[0]),3))) for x in range(len(twistArr))]
    crossGrad = [torch.tensor(numpy.zeros((len(twistArr[0]),3))) for x in range(len(twistArr))]
    vertex_to_edge = vertexToEdge(nf)
    second_order_edge = secondOrderEdge(vertex_to_edge)
    repelIndices = getRepelIndices(second_order_edge)
    #blacklist = [6144, 7174, 7691, 8219, 5665, 5667, 7222, 7739, 8267, 5713, 5715, 7270, 7787, 8315, 5761, 130, 5763, 134, 138, 142, 146, 7318, 150, 154, 7835, 158, 162, 166, 170, 8363, 174, 5809, 178, 5811, 182, 186, 190, 7366, 7883, 8411, 5857, 5859, 7414, 7931, 259, 5380, 263, 267, 271, 5905, 275, 5907, 6934, 279, 283, 287, 291, 7462, 295, 7979, 299, 303, 5425, 307, 5427, 311, 315, 319, 5953, 5955, 6982, 7510, 8027, 5473, 5475, 6001, 6003, 7030, 7558, 8075, 5521, 5523, 6049, 6051, 7078, 7606, 8123, 5569, 5571, 6097, 6099, 7126, 7654, 8171, 5617, 5619]   
    #for i in range(200):
        #lGrad = [getLocalGradient(cPos,indices,svecs) for cPos in cArr]
        #rGrad = [getRepelGradient(cPos,repelIndices,cArr[0]) for cPos in cArr]
        #r = relaxTensor(lGrad,cArr,cGrad,0.0,crossGrad,0.0,cGrad,indices,repelIndices)
        #cGrad = r[0]
        #crossGrad = r[1]
        #relaxElasticEnergyTensor(cArr,indices,[derivA1,derivB1,derivC1])
        #print(('done',i))
        #print(cArr[0][0])
        #if i < 100:
        #    harmonic(cArr,vertex_to_edge,0.995,[])
        #else:
        #    harmonic(cArr,vertex_to_edge,0.995,blacklist)'''
    #for i in range(5):
    #    cArr = harmonic(cArr,vertex_to_edge,0.9)
    #twistArr = cArr
    posList = []
    for x in twistArr:
        nd = dict()
        for y in range(len(x)):
            nd[y] = x[y]
        posList.append(nd)

    '''for x in range(len(twistArr)-1):
        q = isSingular(posList[x],posList[x+1],edge_to_face,joints,mf[4])
        print(x,q)
        q = isSingular2(posList[x],posList[x+1],triangle_to_vertex,mf[4],singularTriangles1)
        print(x,q)
        print('########################################################################################################')
    '''
                
    temp_str = 'var coordDicts = $pos1; var faces = $faces;'
    reorder_faces(nf,pos_list)
    #print(mf[0])
    temp_str = temp_str.replace('$faces',str(nf))
    
    pos_list = []
    for y in twistArr:
        subd=dict()
        for x in range(len(y)):
            subd[x] = list(numpy.copy(y[x]))
            subd[x] = [float(str(y)) for y in subd[x]]
        pos_list.append(subd)
    #print(subd)
    pos_list = round_floats(pos_list)
    temp_str = temp_str.replace('$pos1',str(pos_list))

    #print(interp_type)

    import time
    s = str(int(time.time()))

    g = open('./dataList.js','w')
    g.write(temp_str)
    g.close()

    g = open('./positions.js','w')
    g.write(str(pos_list))
    g.close()

    g = open('./faces.js','w')
    g.write(str(nf))
    g.close()
