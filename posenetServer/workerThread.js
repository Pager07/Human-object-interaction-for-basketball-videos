var workerpool = require('workerpool');
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
var pixels = require('image-pixels');
const {createCanvas, Image} = require('canvas')



var isModelLoaded = false
var net = null
loadModel = async function(req,res){
    console.log("Loading model");
    net = await posenet.load({
        architecture: 'ResNet50',
        outputStride: 32,
        inputResolution: { width: 300 , height: 300 },
        quantBytes: 4,
        multiplier:1
        });
    console.log("Model Loaded!");
    
}


const threadFunction = async (imgBase64Index)=>{
    
    const img = new Image();
    imgBase64Index = imgBase64Index.split('index')
    img.src = imgBase64Index[1]
    const imgIndex = imgBase64Index[0]


    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const input = tf.browser.fromPixels(canvas);
    const imageScaleFactor = 0.3;
    const flipHorizontal = false;
    const outputStride = 8;

    const poses = await net.estimateMultiplePoses(input, {
        flipHorizontal: false,
        maxDetections: 2,
        minPoseConfidence: 0.15,
        minPartConfidence:0.1,
        nmsRadius:20,
        });
    
    boundingBoxes = [] //reset the boundingBoxese
    poses.forEach(pose => {
        var box = posenet.getBoundingBoxPoints(pose.keypoints);
        boundingBoxes.push(box)
    });  
    

    data = [{imgIndex : imgIndex , poses: poses , bbox : boundingBoxes}]
    return data
}

workerpool.worker({
    threadFunction: threadFunction,
    loadModel : loadModel
    });