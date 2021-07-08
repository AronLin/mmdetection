_base_ = './faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    type='FasterRCNN',
    roi_head=dict(bbox_head=dict(type='Shared2FCBBoxHead', num_classes=3)),
)
classes = ('1', '2', '3')

data_root = 'data/myftp/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/',
    ),
    val=dict(
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'test/',
    ),
    test=dict(
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'test/',
    ))

load_from = '/home/PJLAB/linguangchen/work_dir/faster_rcnn/' \
            'faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'
