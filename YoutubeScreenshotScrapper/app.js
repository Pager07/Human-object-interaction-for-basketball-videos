const express = require('express')
const app = express()
const puppeteer = require('puppeteer')
const port = process.env.PORT || 8080
const validUrl = require('valid-url')
const bodyParser = require('body-parser');

app.use(bodyParser.json({limit: '10mb', extended: true}))
app.use(bodyParser.urlencoded({limit: '10mb', extended: true}))
process.setMaxListeners(Infinity); // <== Important line


let page = null;
let browser = null;
const username = 'sunthapa1@hotmail.com';
const password = 'Pager07isgreat';
var linksArr = null; 
var labelsArr = null; 

const login =  async () => {
  browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] })
  page = await browser.newPage()
  await page.setExtraHTTPHeaders({
      'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8'
  });
  await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36')


  await page.setViewport({ width: 1920, height: 1080 })
  await page.goto('https://youtube.com')
  await page.click('#buttons > ytd-button-renderer > a')
  const navigationPromise = page.waitForNavigation()


  await navigationPromise
  await page.waitForSelector('input[type="email"]')
  await page.type('input[type="email"]', username)
  //console.log("Now pressing Enter")
  await page.keyboard.press("Enter");
  //console.log("Pressed Enter!")
  //await page.click('#identifierNext')
  //console.log("Entering Password")
  await page.waitForSelector('input[type="password"]', { visible: true })
  await page.type('input[type="password"]',password)
 //console.log("Password entered!")
  await page.waitForSelector('#passwordNext', { visible: true })
  await page.click('#passwordNext')

  //await navigationPromise

  await page.waitForSelector('#identity-prompt-account-list > ul > label:nth-child(1) > li > span > span.yt-uix-form-input-radio-container > input')
  await page.click('#identity-prompt-account-list > ul > label:nth-child(1) > li > span > span.yt-uix-form-input-radio-container > input')
  await page.click('#identity-prompt-confirm-button')

  await navigationPromise;
 // console.log("Logged Into Youtube.com")

  
}



var parseUrl = function(url) {
  url = decodeURIComponent(url)
  if (!/^(?:f|ht)tps?\:\/\//.test(url)) {
    url = 'http://' + url
  }

  return url
}

app.post('/postLinks' , async function(req,res){
   let linkString = req.body.links
   linksArr = linkString.split(',')
   linksArr.pop()
   //console.log(linksArr);
   res.send('200')
})
app.post('/postLabels' , function(req,res){
  let labelString = req.body.labels;
  labelsArr = labelString.split('/');
  labelsArr.pop();
  res.send('200')
})

app.get('/downloadImages', async function(req, res) {
  // if (req.query.url === undefined || req.query.t === undefined) {
  //   res.send('Invalid url: ' + urlToScreenshot + '&t=' + req.query.t)
  //   return
  // }

  const proccessALink = async(link, label) => {
    //set download path 
    const downloadPath = '/Volumes/My Passport/FinalYearProjectData/ActionClassification/val/' + label;
    //parse the url
    const urlToScreenshot = parseUrl(link)
    //Give a URL it will take a screen shot 
    if (validUrl.isWebUri(urlToScreenshot)) {
      // console.log('Screenshotting: ' + urlToScreenshot + '&t=' + req.query.t)
      console.log('Screenshotting: ' + link);
      //Logic to login to youtube below
      //await login();
      //go to the url and wait till all the content is loaded.
      await page.goto(link, {
        waitUntil: 'networkidle'
        //waitUntil: 'domcontentloaded'
      })
      //await page.waitForNavigation();

      //Find the video player in the page 
      const video = await page.$('.html5-video-player')
      await page.content();

      //Run some command on consoleDev 
      await page.evaluate(() => {
        // Hide youtube player controls.
        let dom = document.querySelector('.ytp-chrome-bottom')
        if (dom != null) {
          dom.style.display = 'none'
        }
      })

      await video.screenshot({
        path: downloadPath
      });
    } else {
      res.send('Invalid url: ' + urlToScreenshot)
    }

  }
  await login();
  var continueDownload = true;
  for(let i = 0; i<labelsArr.length; i++){
    let link = linksArr[i];
    // if(link == 'https://www.youtube.com/watch?v=pxwnayuJtGY&t=40m19s'){
    //   continueDownload = true;
    //   console.log("Starting to download")
    // }
    if(continueDownload){
      let label = labelsArr[i];
      await proccessALink(link, label)
    }
  }
 console.log("Complete!")

res.send('200')
})

app.listen(port, function() {
  console.log('App listening on port ' + port)
})


