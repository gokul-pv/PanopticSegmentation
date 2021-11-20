# Panoptic Segmentation and End-to-End Object Detection using DEtection TRansformer

## Objective

The aim of this project is to train [DETR](https://github.com/facebookresearch/detr) on a custom dataset consisting of objects from construction domain (around 48 classes) for Object Detection and Panoptic Segmentation.

Let us now understand how DETR works and try to answer few questions.

- [Understanding DETR](./Part1/DetrExplanation.md)
- [Points to consider for Panoptic segmenation training](./Part1/README.md)



## Dataset preparation

[Click here](./DatasetCreation/README.md)

## Training

First the object detection model was trained for 200 epochs using pre-trained weights. Then a panoptic head was added on top of this and trained for another 50 epochs. This time the object detection model was freezed and only panoptic head was trained.  

We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone. Horizontal flips, scales and crops are used for augmentation. Images are rescaled to have min size 800 and max size 1333. The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

- Fine-tuning of DETR on construction dataset for Object Detection ([click here](./Detection/README.md))

- Panoptic segmentation training ([click here](./Segmentation/README.md))

## Results

Bounding box detection evaluation results for the construction dataset after training for 200 epochs

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.753
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.864
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.801
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.609
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.782
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.716
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.857
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.871
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.728
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.899
```

Segmentation Metric: (Panoptic, Segmentation, Recognition Quality) after training panoptic head for 50 epochs

```
          |    PQ     SQ     RQ     N
--------------------------------------
All       |  53.1   80.0   60.7    61
Things    |  61.6   82.9   69.6    46
Stuff     |  27.0   71.2   33.5    15
```

Check out the below YouTube link below to see predictions from the trained model

[![Link](https://img.youtube.com/vi/fPUkFKF6qb0/0.jpg)](https://youtu.be/fPUkFKF6qb0)

## Conclusion

The project shows that fine tuning can lead to a score of 53 PQ in about 50 epochs. The results are satisfactory. Transformers are good in global reasoning but are computational expensive with long inputs (high resolution images), making difficult to attain good results with small objects. 

Further works include

- explore new image augmentation techniques like RICAP for better detection results
- reduce leakage of orginial COCO class while creating ground truth. (eg: red areas around wheel loader in [image](./Images/wheel_loader_9996.png))
- add few images from COCO dataset so that PQ for stuff could be increased
- Implement **Spatially Modulated Co-Attention** (SMCA) which is a plug and play module to replace and help achieve faster convergence. Refer this [link](https://arxiv.org/pdf/2108.02404v1.pdf)
- Explore and implement this paper from [Google](https://ai.googleblog.com/2021/04/max-deeplab-dual-path-transformers-for.html) which would allow to skip the BBox detection and directly train for Panoptic segmentation.


## References

1. [DETR Paper](https://arxiv.org/abs/2005.12872)
2. [https://www.youtube.com/watch?v=utxbUlo9CyY](https://www.youtube.com/watch?v=utxbUlo9CyY)
3. [https://wandb.ai/veri/detr/reports/DETR-Panoptic-segmentation-on-Cityscapes-dataset--Vmlldzo2ODg3NjE](https://wandb.ai/veri/detr/reports/DETR-Panoptic-segmentation-on-Cityscapes-dataset--Vmlldzo2ODg3NjE)
4. [https://opensourcelibs.com/lib/finetune-detr](https://opensourcelibs.com/lib/finetune-detr)
5. [https://www.youtube.com/watch?v=RkhXoj_Vvr4](https://www.youtube.com/watch?v=RkhXoj_Vvr4)
6. Explanation on Paper by [Yanic](https://www.youtube.com/watch?v=T35ba_VXkMY) , [AI Epiphany](https://www.youtube.com/watch?v=BNx-wno-0-g)

