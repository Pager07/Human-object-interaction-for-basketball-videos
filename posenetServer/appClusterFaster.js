var cluster = require('cluster')
var numCPUs = require('os').cpus().length;
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
var express = require('express');
var bodyParser = require('body-parser');
var pixels = require('image-pixels');
var async = require("async");


var boundingBoxes = [];
var poseDetectionResults = {}
const {
    createCanvas, Image
} = require('canvas')

var app = express();
app.use(bodyParser.urlencoded({limit: '10mb', extended: true}))

app.get('/getImage',function(req,res){
    var data = JSON.stringify({name:'Sandeep'});
    res.send(data);

})
        

        
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
loadModel()

// app.get('/loadModel', async function(req,res){
//     console.log("Loading model");
//     net = await posenet.load({
//         architecture: 'ResNet50',
//         outputStride: 32,
//         inputResolution: { width: 300 , height: 300 },
//         quantBytes: 4,
//         multiplier:1
//         });
//     console.log("Model Loaded!");
//     res.send("Model Loaded");
    
// })


app.post('/postImage' ,async function(req,res){
    const imgBase64List = req.body.imgBase64.split('next');
    const imgBase64IndexList = []
    imgBase64List.map((imgBase64,index)=>{
        imgBase64IndexList.push([index,imgBase64])
    })
    async.each(imgBase64IndexList , getPoseDetection , function(err){
        if(err){
            console.log("Error occured when feteching one of the pose")
            console.log(err)       
        }else{
            console.log('Sending Poses!')
            res.send(JSON.stringify(poseDetectionResults))
        }
        
    })
        
    

});//End of app.post and asyn function

app.post('/getBBox' ,async function(req,res){
    res.send(JSON.stringify({bbox:boundingBoxes}))
    
});//End of app.post and asyn function



const getPoseDetection = async (imgBase64Index, callback) => {
    //console.log('start');

    const img = new Image();
    img.src = imgBase64Index[1];
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
  
    poseDetectionResults[imgIndex] = {detectionList:poses}
    //console.log(poses)

}
app.listen(3000);






