"""Metric raised-wire geometry, independent of renderer texture and image truth.

World axes: x right, y up, z toward the player. The painted board is z=0.
Camera centre and wire dimensions are explicit assumptions, never inferred truth.
"""
from pathlib import Path
from functools import lru_cache
import cv2 as cv
import numpy as np


def apparent_plane(points,camera_center):
    """Where an elevated point's camera ray intersects the board plane."""
    p=np.atleast_2d(np.asarray(points,np.float64));C=np.asarray(camera_center,np.float64)
    if C.shape!=(3,) or C[2]<=0 or np.any(p[:,2]>=C[2]):raise ValueError('Camera must be in front of all model points')
    return C[:2]+C[2]/(C[2]-p[:,2,None])*(p[:,:2]-C[:2])


def surface_footpoint(apparent_xy,height_mm,camera_center):
    """Recover the elevated feature's x/y, not the incorrect ray/plane hit."""
    q=np.atleast_2d(np.asarray(apparent_xy,np.float64));C=np.asarray(camera_center,np.float64)
    h=np.broadcast_to(np.asarray(height_mm,np.float64),len(q))
    if C[2]<=0 or np.any(h>=C[2]):raise ValueError('Invalid feature/camera height')
    return C[:2]+(C[2]-h[:,None])/C[2]*(q-C[:2])


def mesh(board,height_mm=1.,width_mm=.8,segments=240):
    """Triangular-prism spider: two sloping faces, base at z=0 and apex at z=h."""
    if height_mm<=0 or width_mm<=0:raise ValueError('Positive wire height and width required')
    vertices=[];faces=[]
    paths=[]
    for radius in (6.35,15.9,*board.radii):
        a=np.linspace(0,2*np.pi,segments+1)
        points=np.c_[radius*np.sin(a),radius*np.cos(a)]
        normal=np.c_[np.sin(a),np.cos(a)]
        paths.append((points,normal))
    for spoke in range(20):
        a=np.deg2rad(-9+18*spoke);direction=np.array([np.sin(a),np.cos(a)])
        points=np.array([15.9,board.radii[-1]])[:,None]*direction
        normal=np.tile([direction[1],-direction[0]],(2,1))
        paths.append((points,normal))
    for points,normal in paths:
        base=len(vertices)
        for point,n in zip(points,normal):
            vertices.extend([np.r_[point-width_mm/2*n,0.],np.r_[point,height_mm],np.r_[point+width_mm/2*n,0.]])
        for i in range(len(points)-1):
            a=base+3*i;b=a+3
            faces.extend([(a,b,b+1),(a,b+1,a+1),(a+1,b+1,b+2),(a+1,b+2,a+2)])
    return np.asarray(vertices),np.asarray(faces,int)


def draw_mesh(image,vertices,faces,projected,camera_center):
    """Paint depth-sorted prism faces; geometric, not photorealistic rendering."""
    triangles=vertices[faces];centres=triangles.mean(axis=1)
    order=np.argsort(np.linalg.norm(centres-np.asarray(camera_center),axis=1))[::-1]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-9)
    light=np.array([-.3,.5,1.]);light/=np.linalg.norm(light)
    brightness=np.uint8(np.clip(115+75*abs(normals@light),90,215))
    for f in order:
        polygon=np.rint(projected[faces[f]]*256).astype(np.int32)
        value=int(brightness[f]);colour=value if image.ndim==2 else (value,value,value)
        cv.fillConvexPoly(image,polygon,colour,cv.LINE_AA,shift=8)
    return image


def canonical_template(board,geometry):
    """Board-plane appearance of the raised spider for this view, without K.

    The plane homography accounts for perspective. Camera CENTRE and known
    feature heights account for the difference between z=0 and the raised mesh.
    """
    if not geometry or geometry.get('height_mm',0)<=0:
        from dartboard import ideal_board_template
        return ideal_board_template(board)
    return _template_cached(board.size,tuple(board.radii),tuple(geometry['camera_center_mm']),
                            float(geometry['height_mm']),float(geometry.get('width_mm',.8)))


@lru_cache(maxsize=16)
def _template_cached(size,radii,centre,height,width):
    from dartboard import Board,ideal_board_template
    board=Board(size,radii);image=ideal_board_template(board,wires=False)
    C=np.asarray(centre,np.float64)
    vertices,faces=mesh(board,height,width)
    xy=apparent_plane(vertices,C)
    canonical=board.center+board.ppm*xy*np.array([1.,-1.])
    draw_mesh(image,vertices,faces,canonical,C)
    image.flags.writeable=False
    return image


def elevated_landmarks(board,geometry):
    points=board.landmarks().astype(np.float64)
    if not geometry or geometry.get('height_mm',0)<=0:return points
    xy=(points-board.center)/board.ppm*np.array([1.,-1.])
    q=apparent_plane(np.c_[xy,np.full(len(xy),geometry['height_mm'])],geometry['camera_center_mm'])
    return board.center+board.ppm*q*np.array([1.,-1.])


def export_obj(path,vertices,faces):
    lines=['# Raised dartboard spider, coordinates in millimetres; board surface z=0']
    lines.extend('v '+' '.join(f'{v:.6f}' for v in p) for p in vertices)
    lines.extend('f '+' '.join(str(int(v)+1) for v in f) for f in faces)
    Path(path).write_text('\n'.join(lines)+'\n')
