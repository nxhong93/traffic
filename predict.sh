torch>=1.4.0
timm>=0.3.0
torchvision
pyyaml
numpy
omegaconf
ensemble_boxes
pycocotools>=2.0.2


Train: python train.py --output_path '{Path đến chỗ save model sau khi train}'
default: python train.py --output_path weight/


Test: python predict.py --test_image '{Path image folder tập test}' --result_folder '{Nơi lưu file .submission.json}' --save_image '{Nơi lưu image predict}'
Default: python predict.py --test_image data/ --result_folder result/ --save_image save_image/