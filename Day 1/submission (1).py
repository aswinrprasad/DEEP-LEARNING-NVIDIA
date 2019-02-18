import caffe
import cv2
import sys



def deploy(img_path):
    
    MODEL_JOB_DIR = '/dli/data/digits/20190215-104236-695b'
    get_ipython().system(u'ls $MODEL_JOB_DIR')

    ARCHITECTURE = MODEL_JOB_DIR + '/' + 'deploy.prototxt'
    WEIGHTS = MODEL_JOB_DIR + '/' + 'snapshot_iter_735.caffemodel'
    caffe.set_mode_gpu()
    
    # Initialize the Caffe model using the model trained in DIGITS. Which two files constitute your trained model?
    net = caffe.Classifier(ARCHITECTURE, WEIGHTS,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
                       
    # Create an input that the network expects. This is different for each project, so don't worry about the exact steps, but find the dataset job directory to show you know that whatever preprocessing is done during training must also be done during deployment.
    input_image= caffe.io.load_image(img_path)
    input_image = cv2.resize(input_image, (256,256))
    mean_image = caffe.io.load_image('/dli/data/digits/20190215-103301-c62a')
    input_image = input_image-mean_image
    
    DATA_JOB_DIR = '/dli/data/digits/20190215-104236-695b'  ## Remember to set this to be the job directory for your model
    get_ipython().system(u'ls $DATA_JOB_DIR')
    
    # Make prediction. What is the function and the input to the function needed to make a prediction?
    prediction = net.predict([input_image]) 
    
    if prediction.argmax()==0:
        return !python submission.py '/dli/data/whale/data/train/not_face/w_1.jpg'
    else:
        return !python submission.py '/dli/data/whale/data/train/face/w_1.jpg'  #This should return "whale" at the very bottom

    
##Ignore this part    
if __name__ == '__main__':
    print(deploy(sys.argv[1]))

