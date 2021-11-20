## DETR (Detection Transformer)

The Detection Transformer was developed by the Facebook Research team and introduced in [[1]](http://arxiv.org/abs/1906.05909). DETR works by generating a **set** of features from an input image, and then using a transformer to predict the location (surrounding bounding box) of predefined number of objects as well as their respective class. Each component of the model is explained in detail below. 



![](https://github.com/gokul-pv/EVA6_Assignments_Session14/blob/main/Images/detr_1.png)



The DETR model consists of a pretrained **CNN backbone**, which produces a set of lower dimensional set of features. These features are then scaled and added to a positional encoding, which is fed into a **Transformer** consisting of an **Encoder** and a **Decoder** in a manner quite similar to the Encoder-Decoder transformer described in [[2]](http://arxiv.org/abs/1706.03762). The output of the decoder is then fed into a fixed number of **Prediction Heads** which consist of a predefined number of feed forward networks. Each output of one of these prediction heads consists of a **class prediction**, as well as a predicted **bounding box**. The loss is calculated by computing the bipartite matching loss.



> From the paper:
>
> *We present a new method that views object detection as a direct set  prediction problem. Our approach streamlines the detection pipeline,  effectively removing the need for many hand-designed components like a  non-maximum suppression procedure or anchor generation that explicitly  encode our prior knowledge about the task.* 
>
> *The main ingredient of the new framework, called DEtection TRansformer or  DETR, are a set-based global loss that forces unique predictions via  bipartite matching, and a transformer encoder-decoder architecture.  Given a fixed set of learned object queries, DETR reasons about the  relations of the objects and the global image context to directly output the final st of predictions in parallel. The new model is conceptually  simple and does not require a specialized library, unlike many other  modern detectors.* 
>
> *DETR demonstrates accuracy and run-time performance on par with the  well-established and highly-optimized Faster RCNN baseline on the  challenging COCO object detection dataset. Moreover, DETR can be easily  generalized to produce panoptic segmentation in a unified manner. We  show that it significantly outperforms competitive baselines.*



## CNN Backbone



Assume that our input image is an RGB image with height H0, width W0 and 3 color channels. The CNN backbone consists of a (pretrained) CNN, which we use to generate C lower dimensional features having width W and height H (**In practice, we set C=2048, W=W0/32 and H=H0/32**). Let us assume 640x640x3 as the image size for our discussions. So, the backbone gives us a 20x20x2048 activation map.

This leaves us with C two-dimensional features, and since we will be passing these features into a transformer, each feature must be reformatted in a way that will allow the encoder to process each feature as a sequence. This is done by flattening the feature matrices into an H x W vector, and then concattenating each one.

The flattened convolutional features are added to a spatial positional encoding which can either be learned, or pre-defined.



![](https://github.com/gokul-pv/EVA6_Assignments_Session14/blob/main/Images/detr_2.png)



## Transformer Architecture



![](https://github.com/gokul-pv/EVA6_Assignments_Session14/blob/main/Images/detr_3.png)



**Encoder**

The Encoder consists of N **Encoder Layers**. Each encoder layer consits of a Multi-Head Self-Attention Layer, an **Add & Norm** Layer, a Feed Forward Neural Network, and another **Add & Norm** layer. This is nearly identical to the original Transformer Encoder from [[2]](http://arxiv.org/abs/1706.03762) except we are only adding our spatial positional encoding to the Key and Queue matrices. Also note that we add the spatial encoding tho the Query matrix of the decoder after the decoder's first MHSA and Normalization layer. 



**Decoder**

The decoder is more complicated than the Encoder. The **object queries** consist of a set of N vectors which are added to the key and query matrices of the decoder. The output of the encoder and the spatial positional encoding is added to the key matrix (before the Multi-Head Attention layer). 



**Object Queries**

These are N learnt positional embeddings passed in as inputs to the decoder. These are the N = 100 learned (***nn.Parameter***) vectors/encodings, that finally result in 100 bounding boxes. (DETR can at max detect 100 boxes). These 100 vectors/encodings, are somewhat similar to Anchor Boxes, but unlike them, these are not learning the sizes, but the location of these boxes. These N (100) object queries are transformed into an output embedding by the decoder. They are independently decoded into box coordinates and class labels by a  feed-forward network, resulting in N (100) final predictions.

After training when we plot the centroid of these 100 queries (20 out of 100), we see a pattern:



![](https://github.com/gokul-pv/EVA6_Assignments_Session14/blob/main/Images/ObjectQueries.png)



From the above plot we can see that each of the object query is focusing on a particular region, like the first query is more focused to the bottom left of image and the next  query is looking at the middle part. Since the quarries can communicate with each other due to attention mechanism, each of them communicate with one another and focus on a particular region.These quarries are independent of the class. The decoder updates these embeddings through  multiple self-attention and encoder-decoder attention layers



**Prediction Heads**

The prediction heads consists of two Feed-Forward networks which compute class predictions and bounding boxes. Note that the number of predictions is equal to the number of object queries. An important thing to note here is that If there are less predictions than the number of object queries, then the outputted class will be **no object class**. Each prediction is a tuple containing class and bounding box (c,b).



## Bipartite Matching Loss



Let us assume that we have 2 objects in our image. We expand our ground  truth to 100 objects by added 98 no objects to be predicted. But here is the problem



![](https://github.com/gokul-pv/EVA6_Assignments_Session14/blob/main/Images/Cows.jpg)



We have the top 2 rows for 2 cows, but there is no guarantee that the top 2 in the prediction would be cows (as each query focuses on few  selected/learned areas). This is where **bipartite loss** comes into the picture. We compare our first Ground truth with all the predictions from the Predictions, and select the closest one (so that the loss is less):



![](https://github.com/gokul-pv/EVA6_Assignments_Session14/blob/main/Images/loss_1.png)



This is called **bipartite matching**. This can be found using the **Hungarian Algorithm**. By performing **one-to-one** matching, it is able to significantly reduce  low-quality predictions, and achieve eliminations of output reductions  like NMS. Overall the loss is defined as:



![](https://github.com/gokul-pv/EVA6_Assignments_Session14/blob/main/Images/loss_2.png)



i.e overall loss is the sum of class prediction loss and bounding box difference loss. The loss function for class label is negative log-likelihood. Bounding box loss is a linear combination of L1 loss and Generalized IoU loss to ensure loss is  scale-invariant since there could be small and big boxes.
