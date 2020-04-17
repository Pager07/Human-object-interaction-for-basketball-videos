var numCPUs = require('os').cpus().length;

var express = require('express');
var bodyParser = require('body-parser');
var workerpool = require('workerpool');

var app = express();
app.use(bodyParser.urlencoded({limit: '10mb', extended: true}))

app.get('/getImage',function(req,res){
    var data = JSON.stringify({name:'Sandeep'});
    res.send(data);

})


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

var pool = workerpool.pool(__dirname + '/workerThread.js' );

var loadWorkersModel = async ()=>{
    var promises = Array(7).fill().map((_,index)=>{
        return pool.exec('loadModel')
    })
    //console.log("Loading Models In All Workers!")
    var res = await Promise.all(promises)
    //console.log("Model Loaded In All Workers!")

}

loadWorkersModel()

app.post('/postImage' ,async function(req,res){
    const imgBase64List = req.body.imgBase64.split('next');
    const imgBase64IndexList = []
    imgBase64List.map((imgBase64,index)=>{
        imgBase64IndexList.push([index+'index'+imgBase64])
    })
    var promises = Array(imgBase64IndexList.length).fill().map((_,index)=>{
        
        return pool.exec('threadFunction',imgBase64IndexList[index])
    })
    const segmentsResults = await Promise.all(promises);
    res.send(JSON.stringify({detection:segmentsResults}))
        
    

});//End of app.post and asyn function

app.post('/getBBox' ,async function(req,res){
    res.send(JSON.stringify({bbox:boundingBoxes}))
    
});//End of app.post and asyn function


app.listen(3000);












