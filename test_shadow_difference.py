import unittest
import cv2 as cv
import numpy as np
from dartboard import Board,change_mask,gray_image
from dart_detection import detect_dart
from shadow_difference import shadow_aware_masks


class ShadowDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.board=Board(size=300);self.valid=np.full((300,300),255,np.uint8)
        self.background=np.full((300,300,3),180,np.uint8)
        self.field=np.zeros((300,300),np.float32)
        cv.ellipse(self.field,(150,150),(75,40),25,0,360,1.,-1)
        self.field=cv.GaussianBlur(self.field,(0,0),8)
        self.after=np.uint8(self.background*(1-.23*self.field[...,None]))

    def test_common_soft_shadow_does_not_create_a_foreground(self):
        ordinary=change_mask(gray_image(self.after),gray_image(self.background),self.valid,28)
        self.assertGreater(np.count_nonzero(ordinary),1000)
        masks,_=shadow_aware_masks([self.after]*3,[self.background]*3,[self.valid]*3,self.valid)
        self.assertEqual(sum(np.count_nonzero(m) for m in masks),0)

    def test_neutral_dart_survives_without_hue_change(self):
        images=[];tip=np.array([150,150])
        for end in ((60,90),(245,100),(145,255)):
            image=self.after.copy();cv.line(image,tuple(tip),end,(100,100,100),2,cv.LINE_AA);images.append(image)
        masks,_=shadow_aware_masks(images,[self.background]*3,[self.valid]*3,self.valid)
        point,ids,reason,_,_=detect_dart(masks,self.board)
        # All three neutral shafts remain in the masks. The downstream axis
        # detector may use its supported two-view/review path for a thin raster.
        self.assertTrue(all(cv.countNonZero(m)>150 for m in masks))
        self.assertIsNotNone(point,reason);self.assertGreaterEqual(len(ids),2)
        self.assertLess(np.linalg.norm(point-tip),2.)

    def test_hue_wrap_and_missing_view(self):
        old=np.full((300,300,3),(1,180,200),np.uint8)
        new=np.full((300,300,3),(179,180,155),np.uint8)
        old=cv.cvtColor(old,cv.COLOR_HSV2BGR);new=cv.cvtColor(new,cv.COLOR_HSV2BGR)
        masks,_=shadow_aware_masks([new]*3,[old]*3,[self.valid,self.valid,np.zeros_like(self.valid)],self.valid)
        self.assertEqual(sum(np.count_nonzero(m) for m in masks),0)


if __name__=='__main__':unittest.main()
