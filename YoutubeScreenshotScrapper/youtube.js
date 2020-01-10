const puppeteer = require('puppeteer');
(async () => {
    const browser = await puppeteer.launch({ headless: true })
    const page = await browser.newPage()
    await page.setExtraHTTPHeaders({
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8'
    });
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36')


    await page.setViewport({ width: 1280, height: 800 })
    await page.goto('https://youtube.com')
    await page.click('#buttons > ytd-button-renderer > a')
    const navigationPromise = page.waitForNavigation()


    await navigationPromise
    await page.waitForSelector('input[type="email"]')
    await page.type('input[type="email"]', process.env.GOOGLE_USER)
    console.log("Now pressing Enter")
    await page.keyboard.press("Enter");
    console.log("Pressed Enter!")
    //await page.click('#identifierNext')
    console.log("Entering Password")
    await page.waitForSelector('input[type="password"]', { visible: true })
    await page.type('input[type="password"]',process.env.GOOGLE_PWD)
    console.log("Password entered!")
    await page.waitForSelector('#passwordNext', { visible: true })
    await page.click('#passwordNext')

    //await navigationPromise

    await page.waitForSelector('#identity-prompt-account-list > ul > label:nth-child(1) > li > span > span.yt-uix-form-input-radio-container > input')
    await page.click('#identity-prompt-account-list > ul > label:nth-child(1) > li > span > span.yt-uix-form-input-radio-container > input')
    await page.click('#identity-prompt-confirm-button')

    await navigationPromise;
    console.log("Loaded Youtube.com")

    
})()
