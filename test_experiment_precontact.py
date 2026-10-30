import unittest
import numpy as np

from experiment_precontact import XY_COLUMNS,desired_frames,extract_windows
from extract_uniform_20frames import assess_candidate,candidate_sequences


def fixture(drop_frame=None):
    rows=[]
    for camera,offset in (("C",0),("R",10000)):
        for frame in range(30):
            if frame==drop_frame:continue
            rows.append({"dataset":"EPL","kick_id":"1","camera_direction":camera,"track_id":"1",
                         "frame":str(frame),"last_touch":str(int(frame==25)),"kick_direction":"R",
                         **{column:str(offset+frame*100+i) for i,column in enumerate(XY_COLUMNS)}})
    candidate=next(c for c in candidate_sequences(rows) if c['key'][2]=="C")
    _,record=assess_candidate(candidate,rows)
    ref=np.array([[[float(rows[line-2][column]) for column in XY_COLUMNS]
                   for line in record['selected_source_csv_lines']]],dtype=np.float32)
    return rows,[record],["EPL:1"],ref


class PrecontactTests(unittest.TestCase):
    def test_exact_raw_consecutive_frames_and_endpoint(self):
        self.assertEqual(desired_frames(25,5),[20,21,22,23,24])
        self.assertEqual(desired_frames(25,3),[22,23,24])
        self.assertEqual(desired_frames(25,5,True),[21,22,23,24,25])
        with self.assertRaises(ValueError):desired_frames(25,4)

    def test_source_camera_coordinates_and_labels_preserved(self):
        rows,records,groups,ref=fixture()
        x5,x3,audits,exports=extract_windows(rows,records,groups,ref,False)
        self.assertEqual(x5.shape,(1,5,16));self.assertEqual(x3.shape,(1,3,16))
        np.testing.assert_array_equal(x3,x5[:,-3:])
        self.assertEqual(audits[0]['frames_5'],[20,21,22,23,24])
        for row,line in zip(exports[5],audits[0]['source_csv_lines_5']):
            self.assertEqual({k:row[k] for k in rows[line-2]},rows[line-2])
            self.assertEqual(row['camera_direction'],'C');self.assertEqual(row['kick_direction'],'R')
            self.assertEqual(row['last_touch'],'0')
        np.testing.assert_array_equal(x5[0,:,0],[2000,2100,2200,2300,2400])

    def test_contact_inclusion_is_explicit(self):
        x5,x3,audits,exports=extract_windows(*fixture(),True)
        self.assertEqual(audits[0]['frames_5'],[21,22,23,24,25])
        self.assertEqual(exports[5][-1]['last_touch'],'1')
        np.testing.assert_array_equal(x3,x5[:,-3:])

    def test_missing_raw_frame_does_not_fall_back_to_observed_tail(self):
        with self.assertRaisesRegex(ValueError,'Missing consecutive precontact frames'):
            extract_windows(*fixture(drop_frame=24),False)

    def test_old_sampler_feature_mismatch_rejected(self):
        rows,records,groups,ref=fixture();ref[0,0,0]+=1
        with self.assertRaises(AssertionError):extract_windows(rows,records,groups,ref,False)


if __name__=="__main__":unittest.main()
