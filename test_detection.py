"""Detection behavior tests independent of the renderer's dart pixels."""
import unittest
import cv2 as cv
import numpy as np
from dartboard import Board
from dart_detection import detect_dart,combine_candidates


def axis(point,direction):
    point=np.array(point,float);direction=np.array(direction,float);direction/=np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])
    points=point+np.arange(0,90)[:,None]*direction
    return dict(origin=point,direction=direction,normal=normal,constant=float(normal@point),
                points=points,support_points=points,length_px=90.,rms_px=.2)


class DetectionTests(unittest.TestCase):
    def test_empty_and_broad_clutter_do_not_score(self):
        b=Board();empty=np.zeros((b.size,b.size),np.uint8)
        self.assertIsNone(detect_dart([empty]*3,b)[0])
        blob=empty.copy();cv.ellipse(blob,(450,450),(100,80),30,0,360,255,-1)
        self.assertIsNone(detect_dart([blob]*3,b)[0])

    def test_fragmented_shaft_and_broad_flight(self):
        b=Board();tip=np.array([420.,390.]);masks=[]
        for d in [np.array([1.,.5]),np.array([-1.,.6]),np.array([.1,-1.])]:
            d/=np.linalg.norm(d);n=np.array([-d[1],d[0]])
            mask=np.zeros((900,900),np.uint8)
            for a,z in [(0,20),(35,85),(100,150)]:
                cv.line(mask,tuple(np.rint(tip+a*d).astype(int)),tuple(np.rint(tip+z*d).astype(int)),255,3)
            flight=np.int32([tip+130*d-10*n,tip+175*d,tip+130*d+10*n])
            cv.fillConvexPoly(mask,flight,255);masks.append(mask)
        p,ids,_,_,_=detect_dart(masks,b)
        self.assertEqual(ids,[0,1,2]);self.assertLess(float(np.linalg.norm(p-tip)),1.)

    def test_colour_invisible_tip_in_one_view_is_allowed(self):
        p=[400,350];axes=[[axis(p,[1,.5])],[axis(p,[-1,.5])],[axis(p,[.2,1])]]
        axes[2][0]['support_points']=axes[2][0]['points'][25:]
        self.assertIsNotNone(combine_candidates(axes)[0])

    def test_unsupported_extension_is_rejected(self):
        p=[400,350];axes=[[axis(p,[1,.5])],[axis(p,[-1,.5])],[axis(p,[.2,1])]]
        for c in axes:c[0]['support_points']=c[0]['points'][30:]
        self.assertIsNone(combine_candidates(axes)[0])

    def test_joint_selection_can_choose_a_nonleading_axis(self):
        p=[400,350]
        axes=[[axis([650,100],[1,.8]),axis(p,[1,.5])],
              [axis(p,[-1,.5])],[axis(p,[.2,1])]]
        q,ids,_,selected,_=combine_candidates(axes)
        self.assertEqual(ids,[0,1,2]);np.testing.assert_allclose(q,p,atol=1e-8)
        self.assertIs(selected[0],axes[0][1])

    def test_inconsistent_third_view_does_not_become_a_two_view_score(self):
        p=[400,350]
        axes=[[axis(p,[1,.5])],[axis(p,[-1,.5])],[axis([600,600],[.2,1])]]
        self.assertIsNone(combine_candidates(axes)[0])

    def test_two_different_supported_hits_are_ambiguous(self):
        axes=[[axis([300,300],d),axis([600,600],d)] for d in ([1,.5],[-1,.5],[.2,1])]
        self.assertIsNone(combine_candidates(axes)[0])


if __name__=='__main__':unittest.main()
