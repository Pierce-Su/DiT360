import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from pa_src.i2p_geometry import (intrinsics, rotation, perspective_to_sparse_erp,
                                 erp_to_perspective, padded_known_mask, bilinear)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((81, 101, 3), [23, 79, 201], dtype=np.uint8)
        self.K = intrinsics(101, 81, 80)

    def test_fov_aspect(self):
        self.assertAlmostEqual(self.K[0, 0], self.K[1, 1])
        self.assertAlmostEqual(2*np.arctan(101/(2*self.K[0, 0]))*180/np.pi, 80)
        self.assertLess(2*np.arctan(81/(2*self.K[1, 1]))*180/np.pi, 80)

    def test_identity_bounds_and_polarity(self):
        canvas, known = perspective_to_sparse_erp(self.rgb, self.K, erp_hw=(256, 512))
        self.assertTrue(known[128, 256])
        self.assertFalse(known[:, :128].any())
        self.assertFalse(known[:, 384:].any())
        self.assertFalse(known[:60].any())
        self.assertFalse(known[-60:].any())
        np.testing.assert_array_equal(canvas[~known], np.full((int((~known).sum()), 3), 128))
        np.testing.assert_array_equal(canvas[known], np.tile([23,79,201], (int(known.sum()),1)))
        y, x = np.where(known)
        self.assertLess(abs((x.max()-x.min()+1)/512*360-80), 3)

    def test_off_center_intrinsics(self):
        K = self.K.copy()
        K[0, 2] = 20
        K[1, 1] *= 1.2
        _, known = perspective_to_sparse_erp(self.rgb, K, erp_hw=(256,512))
        self.assertGreater(np.where(known)[1].mean(), 256)

    def test_yaw_seam_and_roundtrip(self):
        _, known = perspective_to_sparse_erp(self.rgb, self.K, rotation(180), (256, 512))
        self.assertTrue(known[:, :20].any() and known[:, -20:].any())
        self.assertFalse(known[:, 220:290].any())
        for yaw in [0, 90, 180, -90]:
            canvas, _ = perspective_to_sparse_erp(self.rgb, self.K, rotation(yaw), (512,1024))
            back = erp_to_perspective(canvas, self.K, rotation(yaw), (81,101))
            np.testing.assert_allclose(back[4:-4, 4:-4], self.rgb[4:-4, 4:-4], atol=1)

    def test_coordinate_ramp_roundtrip(self):
        y, x = np.mgrid[:81,:101]
        rgb = np.stack([x*2,y*3,x+y], axis=-1).astype(np.uint8)
        for R in [rotation(), rotation(180), rotation(35,20)]:
            canvas, _ = perspective_to_sparse_erp(rgb, self.K, R, (512,1024))
            back = erp_to_perspective(canvas, self.K, R, (81,101))
            self.assertLess(np.abs(back[4:-4,4:-4].astype(float)-rgb[4:-4,4:-4]).max(), 2)

    def test_periodic_bilinear(self):
        rgb = np.zeros((2,8,3)); rgb[:,0] = 100; rgb[:,-1] = 200
        for x in [-.5, 7.5, 15.5]:
            np.testing.assert_allclose(bilinear(rgb, np.array([x]), np.array([.5]), True), 150)

    def test_poles(self):
        canvas, _ = perspective_to_sparse_erp(self.rgb, self.K, rotation(0,90), (512,1024))
        back = erp_to_perspective(canvas, self.K, rotation(0,90), (81,101))
        np.testing.assert_allclose(back[4:-4,4:-4], self.rgb[4:-4,4:-4], atol=1)

    def test_mask_padding_flatten_crop(self):
        known = np.zeros((32,64), bool)
        known[:16,:16] = True
        known[16:,48:] = True
        expected = np.array([[0,1,0,0,0,1],[1,0,0,0,1,0]], np.float32)
        mask = padded_known_mask(known).reshape(2,6)
        np.testing.assert_array_equal(mask, expected)
        ramp = np.arange(8).reshape(2,4)
        padded = np.concatenate([ramp[:,-1:],ramp,ramp[:,:1]],1)
        np.testing.assert_array_equal(padded.reshape(-1).reshape(2,6)[:,1:-1], ramp)

    def test_reject_invalid(self):
        for fov in [0,180,-1,float('nan')]:
            with self.assertRaises(ValueError): intrinsics(100,80,fov)
        for K in [np.eye(2), np.diag([-1,1,1]), np.full((3,3),np.nan)]:
            with self.assertRaises(ValueError): perspective_to_sparse_erp(self.rgb,K)
        with self.assertRaises(ValueError): perspective_to_sparse_erp(self.rgb,self.K,np.ones((3,3)))
        with self.assertRaises(ValueError): padded_known_mask(np.zeros((31,64),bool))
        K = self.K.copy(); K[0,2] = 1e10
        with self.assertRaises(ValueError): perspective_to_sparse_erp(self.rgb,K)

    def test_cli_prepare_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); image=root/'source.jpg'; out=root/'out'
            Image.fromarray(self.rgb).save(image)
            command=[sys.executable,'i2p.py','--image',str(image),'--caption','Exact caption.',
                     '--hfov-deg','80','--output-dir',str(out)]
            r=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            data=json.loads((out/'preparation.json').read_text())
            self.assertEqual(data['effective_prompts'],['This is a panorama image. Exact caption.']*2)
            self.assertFalse((out/'raw_erp.png').exists())
            self.assertTrue((out/'PREPARED').exists())
            before=(out/'preparation.json').read_bytes()
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual((out/'preparation.json').read_bytes(),before)
            self.assertNotEqual(subprocess.run(command+['--K-json','none'],capture_output=True).returncode,0)
            self.assertNotEqual(subprocess.run(command+['--generate'],capture_output=True).returncode,0)


if __name__=='__main__': unittest.main()
