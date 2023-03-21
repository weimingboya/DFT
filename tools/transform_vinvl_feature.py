import h5py
import os
import json
import base64
import numpy as np
from tqdm import tqdm
from common.utils.tsv_file import TSVFile

def extract_feature(detections_path):
    save_path = os.path.join('coco/features', 'COCO2014_VinVL_TEST.hdf5')
    f = h5py.File(save_path, mode='w')

    file_feat = TSVFile(os.path.join(detections_path, 'features.tsv'))
    file_label = TSVFile(os.path.join(detections_path, 'predictions.tsv'))

    for i in tqdm(range(file_feat.num_rows())):
        row_feat = file_feat.seek(i)
        row_label = file_label.seek(i)

        assert row_feat[0] == row_label[0]
        image_id = int(row_feat[0].split('_')[-1])

        feature = np.frombuffer(base64.b64decode(row_feat[2]), np.float32)
        feature = feature.reshape((int(row_feat[1]), -1))[:,:-6]

        row_label = json.loads(row_label[1])
        objects = row_label['objects']
        size = np.array([row_label['image_h'], row_label['image_w']])
        boxes = np.array([])
        cls_prob = np.array([])
        if len(objects):
            boxes = np.stack([np.array(l['rect']) for l in objects])
            cls_prob = np.stack([np.array(l['conf']) for l in objects])

        f.create_dataset('%s_features' % image_id, data=feature)
        f.create_dataset('%s_boxes' % image_id, data=boxes)
        f.create_dataset('%s_size' % image_id, data=size)
        f.create_dataset('%s_cls_prob' % image_id, data=cls_prob)

    f.close()

def extract_label(detections_path):
    res = dict()
    cnt = 0

    bi_words = []
    for split in ('train', 'val', 'test'):
        file_label = TSVFile(os.path.join(detections_path, '%s.label.tsv' % split))

        for i in tqdm(range(file_label.num_rows())):
            row_label = file_label.seek(i)
            image_id = row_label[0]

            labels = json.loads(row_label[1])

            arr = []
            s = set()
            for l in labels:
                x = l['class']

                words = x.split()
                if 'and' in words:
                    words.remove('and')
                if '&' in words:
                    words.remove('&')
                if 'Human' in words:
                    continue

                for word in words:
                    if word not in s:
                        s.add(word)
                        arr.append(word)

                if len(x.split()) > 1:
                    bi_words.append(x)

            if len(arr) == 0:
                cnt += 1
                print('image_id: %s' % image_id)
                
            line = ' '.join(arr)
            res[image_id] = line

    with open('coco/features/COCO2014_VinVL_labels.json','w') as fp:
        json.dump(res,fp)

    print('nums of Nill:%s' % cnt)
    print(set(bi_words))

if __name__=='__main__':
    # Please go to https://github.com/pzzhang/VinVL/blob/main/DOWNLOAD.md to download the original VinVL features
    
    detections_path = 'Folder path of the downloaded VinVL features'
    # extract_label(detections_path)
    extract_feature(detections_path)