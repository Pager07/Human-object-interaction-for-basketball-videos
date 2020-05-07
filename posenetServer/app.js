const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
var express = require('express');
var bodyParser = require('body-parser');
var pixels = require('image-pixels');



var boundingBoxes = [];

const {
    createCanvas, Image
} = require('canvas')

var app = express();
app.use(bodyParser.urlencoded({limit: '10mb', extended: true}))

app.get('/getImage',function(req,res){
    var data = JSON.stringify({name:'Sandeep'});
    res.send(data);
})

app.get('/loadModel', async function(req,res){
    console.log("Loading model");
    net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: 300 , height: 300},
        quantBytes: 4,
        multiplier:1
        });
    console.log("Model Loaded!");
    res.send("Model Loaded");
    
})


app.post('/postImage' ,async function(req,res){
   const imgBase64 = req.body.imgBase64;
   const poseKeypoints = await getPoseDetection(imgBase64) 
   res.send(JSON.stringify({detectionList:poseKeypoints}))

});//End of app.post and asyn function

app.post('/getBBox' ,async function(req,res){
    res.send(JSON.stringify({bbox:boundingBoxes}))
    
});//End of app.post and asyn function


// L
// loadModel = async function(req,res){
//     console.log("Loading model");
//     net = await posenet.load({
//         architecture: 'ResNet50',
//         outputStride: 32,
//         inputResolution: { width: 300 , height: 300 },
//         quantBytes: 4,
//         multiplier:1
//         });
//     console.log("Model Loaded!");
    
// }
// loadModel()


//
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
    const imageScaleFactor = 0.3;
    const flipHorizontal = false;
    const outputStride = 8;
    //const pose = await net.estimateMultiplePoses(input, imageScaleFactor, flipHorizontal, outputStride);
    // const pose = await net.estimateMultiplePoses(input, {
    //     flipHorizontal: false,
    //     maxDetections: 5,
    //     scoreThreshold: 0.5,
    //     nmsRadius: 30,
    //     });
    const poses = await net.estimateMultiplePoses(input, {
        flipHorizontal: false,
        maxDetections: 1,
        minPoseConfidence: 0.15,
        minPartConfidence:0.1,
        nmsRadius:20,
        });
    
    boundingBoxes = [] //reset the boundingBoxese
    poses.forEach(pose => {
        var box = posenet.getBoundingBoxPoints(pose.keypoints);
        boundingBoxes.push(box)
    });  
    return poses;
}
app.listen(3000);