"""Soft board-plane shadows from projected dart geometry and sampled area lights.

An ambient floor limits darkness. This approximates area-light occlusion; it is
not a full lighting/path-tracing model and does not shade the dart itself.
"""
import cv2 as cv
import numpy as np


def shadowed_texture(texture,board,darts,lights,seed):
    rng=np.random.default_rng(seed);transmission=np.ones(texture.shape[:2],np.float32)
    for light in lights:
        coverage=np.zeros(texture.shape[:2],np.float32)
        for _ in range(12):
            r=light['radius_mm']*np.sqrt(rng.random());theta=rng.uniform(0,2*np.pi)
            L=np.array(light['position_mm'],float)+[r*np.cos(theta),r*np.sin(theta),0]
            layer=np.zeros(texture.shape[:2],np.uint8)
            def footprint(points):
                p=np.asarray(points,float)
                t=-L[2]/(p[:,2]-L[2])
                q=L+(p-L)*t[:,None]
                return np.rint(board.center+q[:,:2]*board.ppm*np.array([1,-1])).astype(np.int32)
            for dart in darts:
                tip=np.array(dart['tip'],float);direction=np.array(dart.get('direction',[.12,-.18,1.]))
                length=dart.get('length',73.);p=np.array([tip,tip+13/73*length*direction,tip+40/73*length*direction,tip+length*direction])
                xy=footprint(p)
                for index,width in enumerate((.7,2.5,1.1)):
                    cv.line(layer,tuple(xy[index]),tuple(xy[index+1]),255,max(1,round(width*board.ppm)),cv.LINE_AA)
                base=tip+59/73*length*direction;w=dart.get('flight_width',5.)
                flight=footprint([base+[-w,0,0],p[-1],base+[w,0,0],tip+54/73*length*direction])
                cv.fillConvexPoly(layer,flight,255,cv.LINE_AA)
            coverage+=layer.astype(np.float32)/(255*12)
        coverage=cv.GaussianBlur(coverage,(0,0),.8*board.size/900)
        transmission*=1-light['strength']*coverage
    return np.uint8(np.clip(texture.astype(np.float32)*transmission[...,None],0,255)),transmission
