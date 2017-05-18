# express_recognition
## Introduce
In this project, we use deep learning to recognize the province, city and telephone numbers.

## Method
### Province, city and district
In this part, we treat it as a classification problem. In China, there are about 30 province and 150 cities. We use just convolutional neural network to do this task.

### telephone numbers
In this part, we use the method refered to in this [paper](http://arxiv.org/abs/1507.05717). In summary, we use cnn to do feature extraction, and rnn to process feature sequence, and ctc loss to evaluate it. It is a sequence to sequence model.

## Dependencies
We use PyTorch to implement almost all our models.

## Citation

    @article{ShiBY15,
      author    = {Baoguang Shi and
                   Xiang Bai and
                   Cong Yao},
      title     = {An End-to-End Trainable Neural Network for Image-based Sequence Recognition
                   and Its Application to Scene Text Recognition},
      journal   = {CoRR},
      volume    = {abs/1507.05717},
      year      = {2015}
    }

## Acknowledgements
Please let me know if you encounter any issues.
