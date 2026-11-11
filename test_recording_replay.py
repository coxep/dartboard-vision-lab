"""Exercise the offline replay reader with synthetic encoded video files."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import cv2 as cv


class RecordingReplayTests(unittest.TestCase):
    def test_encoded_no_change_dart_and_occlusion_windows(self):
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix='dart-replay-') as directory:
            folder=Path(directory);sources=[]
            for i in range(3):
                baseline=cv.imread(str(root/f'c920_benchmark/camera_{i}_empty.jpg'))
                dart=cv.imread(str(root/f'c920_benchmark/treble20/camera_{i}_input.jpg'))
                occluded=dart.copy();cv.rectangle(occluded,(100,100),(1700,950),(30,50,80),-1)
                path=folder/f'camera_{i}.avi'
                writer=cv.VideoWriter(str(path),cv.VideoWriter_fourcc(*'MJPG'),5,(baseline.shape[1],baseline.shape[0]))
                self.assertTrue(writer.isOpened())
                for image in (baseline,baseline,dart,dart,occluded,occluded):writer.write(image)
                writer.release();sources.append(str(path))
            events=[dict(reference_frames=[0]*3,settled_frames=[1]*3),
                    dict(reference_frames=[0]*3,settled_frames=[3]*3,truth_label='T20',truth_xy_mm=[0,-103]),
                    dict(reference_frames=[0]*3,settled_frames=[5]*3)]
            event_path=folder/'events.json';event_path.write_text(json.dumps(events))
            output=folder/'output'
            run=subprocess.run([sys.executable,str(root/'replay_recordings.py'),'--sources',*sources,
                                '--calibration',str(root/'c920_benchmark/calibration.json'),'--events',str(event_path),
                                '--output',str(output)],capture_output=True,text=True,timeout=90)
            self.assertEqual(run.returncode,0,run.stderr)
            report=json.loads((output/'report.json').read_text())['events']
            self.assertEqual([r['outcome'] for r in report],['rejected','correct','rejected'])
            self.assertLess(report[1]['error_mm'],1.)


if __name__=='__main__':unittest.main()
