$(document).ready(function () {
    console.log("document ready!");
    cv['onRuntimeInitialized'] = () => {
        //function to Calculate diameter of trunk from one dimensional predictions Array with length 224*224 = 50176 of deepLabV3+
//, trained with treeo dataset
// Input: - predictions from deepLabV3+
// Outputs: - 0, if no card or no trunk detected or if predictions has not the size of 50176
//          - Diameter in cm, if card and trunk are detected and if predictions has size of 50176
//REQUIRES OPENCV to work. Can be downloaded from here: https://docs.opencv.org/master/opencv.js
        function evaluateDeeplabPredictions(predictions) {
            let width = 224;
            let height = 224;
            //check if predictions has correct size
            if (predictions.length != width * height) {
                console.log("predictions has wrong size");
                return 0;
            }
            let closingFactor = 20;
            let openingFactor = 20;
            //CREATE PICTURE FROM PREDICTIONS

            let buffer = new Uint8ClampedArray(width * height * 4);
            //save all occurences
            let classifications = [];
            // row after row
            for (let y = 0; y < height; y++) {
                //index after index
                for (let x = 0; x < width; x++) {
                    let classification = predictions[y * width + x];
                    if (!classifications.includes(classification)) {
                        classifications.push(classification);
                    }
                    let pos = (y * width + x) * 4; // position in buffer based on x and y
                    //background
                    if (classification == 0) {
                        buffer[pos] = 0;           // some R value [0, 255]
                        buffer[pos + 1] = 0;                     // some G value
                        buffer[pos + 2] = 0;                     // some B value
                        buffer[pos + 3] = 255;                 // set alpha channel

                    }
                    //trunk
                    else if (classification == 1) {
                        buffer[pos] = 127;           // some R value [0, 255]
                        buffer[pos + 1] = 127;                     // some G value
                        buffer[pos + 2] = 127;                     // some B value
                        buffer[pos + 3] = 255;                 // set alpha channel

                    }
                    //card
                    else if (classification == 2) {
                        buffer[pos] = 255;           // some R value [0, 255]
                        buffer[pos + 1] = 255;                     // some G value
                        buffer[pos + 2] = 255;                     // some B value
                        buffer[pos + 3] = 255;                 // set alpha channel

                    }

                }
            }
            //Check if no card or no trunk Pixel was detected
            if (!classifications.includes(1)) {
                console.log("no trunk pixel detected");
                return 0;
            }
            if (!classifications.includes(2)) {
                console.log("no card pixel detected");
                return 0;
            }

            // create off-screen canvas element
            let canvas = document.createElement('canvas');
            let ctx = canvas.getContext('2d');

            canvas.width = width;
            canvas.height = height;

            // create imageData object
            let idata = ctx.createImageData(width, height);

            // set our buffer as source
            idata.data.set(buffer);
            // update canvas with new data
            ctx.putImageData(idata, 0, 0);


            //orig greyscale masks
            let mat = cv.imread(canvas, 0);
            //binary card mask
            let card = new cv.Mat();
            //binary trunk mask
            let trunk = mat;

            //seperate card mask
            //means every value below 254, will be set to 0, and above 254 to the value of 255
            cv.threshold(mat, card, 254, 255, cv.THRESH_BINARY);

            //seperate trunk mask
            //for trunk its in range from 127 to 127 only
            for (let i = 0; i < trunk.rows; i++) {
                for (let j = 0; j < trunk.cols; j++) {
                    let editValue = trunk.ucharPtr(i, j);
                    if (editValue[0] != 127) //check whether value is within range.
                    {
                        for (let r = 0; r < 3; r++) {
                            trunk.ucharPtr(i, j)[r] = 0;
                        }
                    } else {
                        for (let r = 0; r < 3; r++) {
                            trunk.ucharPtr(i, j)[r] = 255;
                        }
                    }
                }
            }
            //console.log("masks seperated");


            ///get size of CARD
            //1.) RGBA to ONE CHANNEL
            cv.cvtColor(card, card, cv.COLOR_RGBA2GRAY, 0);
            /*
            console.log('card width: ' + card.cols + '\n' +
                'card height: ' + card.rows + '\n' +
                'card size: ' + card.size().width + '*' + card.size().height + '\n' +
                'card depth: ' + card.depth() + '\n' +
                'card channels ' + card.channels() + '\n' +
                'card type: ' + card.type() + '\n');
            */
            //2.) CLOSE OPERATION TO KILL NOISE
            //close and open to kill noise
            let cardCloseFilter = cv.Mat.ones(closingFactor, closingFactor, cv.CV_8U);
            let closedCard = new cv.Mat();
            cv.morphologyEx(card, closedCard, cv.MORPH_CLOSE, cardCloseFilter);
            let cardOpenFilter = cv.Mat.ones(openingFactor, openingFactor, cv.CV_8U);
            let openedCard = new cv.Mat();
            cv.morphologyEx(closedCard, openedCard, cv.MORPH_OPEN, cardOpenFilter);


            //3.) FIND COUNTOURS
            let cardCountoursDrawn = cv.Mat.zeros(card.cols, card.rows, cv.CV_8UC3);
            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.findContours(openedCard, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            //console.log("card contours found");

            // DRAW COUNTOURS
            for (let i = 0; i < contours.size(); ++i) {
                let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                    Math.round(Math.random() * 255));
                cv.drawContours(cardCountoursDrawn, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
            }
            //console.log("card contours drawn");

            //4.) FIND MIN AREA RECT OF CONTOURS
            let cardRect = cv.Mat.zeros(card.rows, card.cols, cv.CV_8UC3);
            let cardRotatedRect = cv.minAreaRect(contours.get(0));
            let cardVertices = cv.RotatedRect.points(cardRotatedRect);
            let cardRectangleColor = new cv.Scalar(255, 0, 0);
            //DRAW MIN AREA RECT OF CONTOURS
            //console.log("card begin draw rectangle");
            for (let i = 0; i < 4; i++) {
                cv.line(cardRect, cardVertices[i], cardVertices[(i + 1) % 4], cardRectangleColor, 2, cv.LINE_AA, 0);
            }

            // 5.) GET SIZE OF CARD_RECTANGLE IN PIXELS
            function getRange(x1, y1, x2, y2) {
                return Math.sqrt(Math.pow((x2 - x1), 2) + Math.pow((y2 - y1), 2));
            }

            //check range to each point from point [0], second most far away is point to longer side
            let range0_to_1 = getRange(cardVertices[0]["x"], cardVertices[0]["y"], cardVertices[1]["x"], cardVertices[1]["y"]);
            let range0_to_2 = getRange(cardVertices[0]["x"], cardVertices[0]["y"], cardVertices[2]["x"], cardVertices[2]["y"]);
            let range0_to_3 = getRange(cardVertices[0]["x"], cardVertices[0]["y"], cardVertices[3]["x"], cardVertices[3]["y"]);
            //get second biggest
            let distances = [range0_to_1, range0_to_2, range0_to_3];
            distances.sort(function (a, b) {
                return a - b
            });
            let cardlongerSide = distances[1];
            let cardshorterSide = distances [0];
            //console.log("cardlong: ", cardlongerSide);
            //console.log("cardshort: ", cardshorterSide);


            ///get size of TRUNK
            //1.) RGBA to ONE CHANNEL
            cv.cvtColor(trunk, trunk, cv.COLOR_RGBA2GRAY, 0);
            /*
            console.log('trunk width: ' + trunk.cols + '\n' +
                'trunk height: ' + trunk.rows + '\n' +
                'trunk size: ' + trunk.size().width + '*' + trunk.size().height + '\n' +
                'trunk depth: ' + trunk.depth() + '\n' +
                'trunk channels ' + trunk.channels() + '\n' +
                'trunk type: ' + trunk.type() + '\n');
            */
            //2.) CLOSE OPERATION TO KILL NOISE AND CONNECT MASKS
            let trunkCloseFilter = cv.Mat.ones(closingFactor, closingFactor, cv.CV_8U);
            let closedTrunk = new cv.Mat();
            cv.morphologyEx(trunk, closedTrunk, cv.MORPH_CLOSE, trunkCloseFilter);
            let trunkOpenFilter = cv.Mat.ones(openingFactor, openingFactor, cv.CV_8U);
            let openedTrunk = new cv.Mat();
            cv.morphologyEx(closedTrunk, openedTrunk, cv.MORPH_OPEN, trunkOpenFilter);

            //3.) FIND COUNTOURS
            let trunkCountoursDrawn = cv.Mat.zeros(card.cols, card.rows, cv.CV_8UC3);
            contours = new cv.MatVector();
            hierarchy = new cv.Mat();
            cv.findContours(openedTrunk, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            //console.log("trunk contours found");
            // DRAW COUNTOURS
            for (let i = 0; i < contours.size(); ++i) {
                let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
                    Math.round(Math.random() * 255));
                cv.drawContours(trunkCountoursDrawn, contours, i, color, 1, cv.LINE_8, hierarchy, 100);
            }
            //console.log("trunk contours drawn");
            //4.) FIND MIN AREA RECT OF CONTOURS
            let trunkRect = cv.Mat.zeros(trunk.rows, trunk.cols, cv.CV_8UC3);
            let trunkRotatedRect = cv.minAreaRect(contours.get(0));
            let trunkVertices = cv.RotatedRect.points(trunkRotatedRect);
            let trunkRectangleColor = new cv.Scalar(255, 0, 0);
            //DRAW MIN AREA RECT OF CONTOURS
            //console.log("card begin draw rectangle");
            for (let i = 0; i < 4; i++) {
                cv.line(trunkRect, trunkVertices[i], trunkVertices[(i + 1) % 4], trunkRectangleColor, 2, cv.LINE_AA, 0);
            }

            // 5.) GET SIZE OF TRUNK_RECTANGLE IN PIXELS
            //check range to each point from point [0], second most far away is point to longer side
            range0_to_1 = getRange(trunkVertices[0]["x"], trunkVertices[0]["y"], trunkVertices[1]["x"], trunkVertices[1]["y"]);
            range0_to_2 = getRange(trunkVertices[0]["x"], trunkVertices[0]["y"], trunkVertices[2]["x"], trunkVertices[2]["y"]);
            range0_to_3 = getRange(trunkVertices[0]["x"], trunkVertices[0]["y"], trunkVertices[3]["x"], trunkVertices[3]["y"]);

            //get second biggest
            distances = [range0_to_1, range0_to_2, range0_to_3];
            distances.sort(function (a, b) {
                return a - b
            });
            let trunklongerSide = distances[1];
            let trunkshorterSide = distances [0];

            //console.log("trunklong: ", trunklongerSide);
            //console.log("trunkshort: ", trunkshorterSide);
            //console.log("cardlong: ", cardlongerSide);
            //console.log("cardshort: ", cardshorterSide);

            //COMPARE SIZES TO ESTIMATE DIAMETER
            let cardLength = 856; //mm
            let pixelSize = cardLength / cardlongerSide;
            let trunkDiameter = (trunkshorterSide * pixelSize) / 100;  //cm

            console.log("EstimatedDiameter: ", trunkDiameter);
            return trunkDiameter;
        }

///TESTING ONLY///
        function test() {
            //blank image
            let testBackground = [];
//trunk mask only
            let testTrunk = [];
//card mask only
            let testCard = [];
//card missing
            let testBackgroundTrunk = [];
//trunk missing
            let testBackgroundCard = [];
//background missing
            let testTrunkCard = [];
//wrong size
            let testWrongSize = [];
// "normal picture"
            let testNormal = [];

//generate Test Predictions
//blank image
            for (let i = 0; i < 50176; i++) {
                testBackground.push(0);
            }
//trunk mask only
            for (let i = 0; i < 50176; i++) {
                testTrunk.push(1);
            }
//trunk mask only
            for (let i = 0; i < 50176; i++) {
                testCard.push(2);
            }
//card missing
            for (let i = 0; i < 50176; i++) {
                if (Math.random() < 0.5) {
                    testBackgroundTrunk.push(0);
                } else {
                    testBackgroundTrunk.push(1);
                }
            }
//trunk missing
            for (let i = 0; i < 50176; i++) {
                if (Math.random() < 0.5) {
                    testBackgroundCard.push(0);
                } else {
                    testBackgroundCard.push(2);
                }
            }
//background missing
            for (let i = 0; i < 50176; i++) {
                if (Math.random() < 0.5) {
                    testTrunkCard.push(1);
                } else {
                    testTrunkCard.push(2);
                }
            }
//Wrong Size  --> 224 pixel= one row missing
            for (let i = 0; i < 49952; i++) {
                if (Math.random() < 0.33) {
                    testWrongSize.push(0);
                } else if (Math.random() < 0.66) {
                    testWrongSize.push(1);
                } else {
                    testWrongSize.push(2);
                }
            }
//normal
            for (let i = 0; i < 50176; i++) {
                if (Math.random() < 0.33) {
                    testNormal.push(0);
                } else if (Math.random() < 0.66) {
                    testNormal.push(1);
                } else {
                    testNormal.push(2);
                }
            }
//run tests
            let fail = 0;
            if (evaluateDeeplabPredictions(testBackground) != 0) {
                console.log("testBackground failed");
                fail = 1;
            }
            if (evaluateDeeplabPredictions(testTrunk) != 0) {
                console.log("testTrunk failed");
                fail = 1;
            }
            if (evaluateDeeplabPredictions(testCard) != 0) {
                console.log("testCard failed");
                fail = 1;
            }
            if (evaluateDeeplabPredictions(testBackgroundTrunk) != 0) {
                console.log("testBackgroundTrunk failed");
                fail = 1;
            }
            if (evaluateDeeplabPredictions(testTrunkCard) == 0) {
                console.log("testTrunkCard failed");
                fail = 1;
            }
            if (evaluateDeeplabPredictions(testBackgroundCard) != 0) {
                console.log("testBackgroundCard failed");
                fail = 1;
            }
            if (evaluateDeeplabPredictions(testWrongSize) != 0) {
                console.log("testWrongSize failed");
                fail = 1;
            }
            if (evaluateDeeplabPredictions(testNormal) == 0) {
                console.log("testNormal failed");
                fail = 1;
            }
            if (!fail) {
                console.log("AllTestSuccessfull!");
            } else {
                console.log("test failed!");

            }

        }

        test();
    };


});




