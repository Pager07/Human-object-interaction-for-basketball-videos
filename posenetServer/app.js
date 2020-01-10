const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
var express = require('express');
var bodyParser = require('body-parser');
var pixels = require('image-pixels');

const {
    createCanvas, Image
} = require('canvas')

var app = express();

app.use(bodyParser.urlencoded({ extended: false }))

app.get('/getImage',function(req,res){
    var data = JSON.stringify({name:'Sandeep'});
    res.send(data);
})

app.get('/loadModel', async function(req,res){
    net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: 257, height: 200 },
        quantBytes: 2
        });
    
})

app.post('/postImage' ,async function(req,res){
   const imgBase64 = req.body.imgBase64;
   const poseKeypoints = await getPoseDetection(imgBase64) 
   res.send(JSON.stringify({detectionList:poseKeypoints}))

});//End of app.post and asyn function

const getPoseDetection = async(imageUrl) => {
    console.log('start');
    // const net = await posenet.load({
    //     architecture: 'ResNet50',
    //     outputStride: 32,
    //     inputResolution: { width: 257, height: 200 },
    //     quantBytes: 2
    //     });
    const img = new Image();
    img.src = imageUrl;
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const input = tf.browser.fromPixels(canvas);
    const imageScaleFactor = 0.5;
    const flipHorizontal = false;
    const outputStride = 16;
    const pose = await net.estimateSinglePose(input, imageScaleFactor, flipHorizontal, outputStride);
    // for(const keypoint of pose.keypoints) {
    //     console.log(`${keypoint.part}: (${keypoint.position.x},${keypoint.position.y})`);
    // }
    return pose;
}
app.listen(3000);