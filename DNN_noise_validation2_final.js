/**************************
 * Dnn_noise_validation2 Experiment *
 **************************/

import { PsychoJS } from './lib_bugfix/core-2020.1.js';
import * as core from './lib_bugfix/core-2020.1.js';
import { TrialHandler } from './lib_bugfix/data-2020.1.js';
import { Scheduler } from './lib_bugfix/util-2020.1.js';
import * as util from './lib_bugfix/util-2020.1.js';
import * as visual from './lib_bugfix/visual-2020.1.js';
import * as sound from './lib_bugfix/sound-2020.1.js';

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0.1, 0.1, 0.1]),
  units: 'pix', ////////// ******
  waitBlanking: true
});

// store info about the experiment session:
let expName = 'DNN_noise_validation2';  // from the Builder filename that created this script
let expInfo = {'participant': '', 'session': '001'};

// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(downloadFiles);
flowScheduler.add(experimentInit);
flowScheduler.add(start_instructionsRoutineBegin());
flowScheduler.add(start_instructionsRoutineEachFrame());
flowScheduler.add(start_instructionsRoutineEnd());
flowScheduler.add(start_instructions2RoutineBegin());
flowScheduler.add(start_instructions2RoutineEachFrame());
flowScheduler.add(start_instructions2RoutineEnd());
flowScheduler.add(query_sizeRoutineBegin());
flowScheduler.add(query_sizeRoutineEachFrame());
flowScheduler.add(query_sizeRoutineEnd());
const blocksLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(blocksLoopBegin, blocksLoopScheduler);
flowScheduler.add(blocksLoopScheduler);
flowScheduler.add(blocksLoopEnd);
const trials_2LoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trials_2LoopBegin, trials_2LoopScheduler);
flowScheduler.add(trials_2LoopScheduler);
flowScheduler.add(trials_2LoopEnd);
flowScheduler.add(endRoutineBegin());
flowScheduler.add(endRoutineEachFrame());
flowScheduler.add(endRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

var n_im_files = 100;
var im_files = new Array(n_im_files);
for (var i = 0; i < n_im_files; i++) {
  im_files[i] = { name: 'Images/valid2_' + i.toString() + '.png', path: 'resources/Images/valid2_' + i.toString() + '.png' };
}

psychoJS.start({
expName: expName,
expInfo: expInfo,
resources: im_files
});

var frameDur;
function updateInfo() {
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2020.1.2';
  expInfo['OS'] = window.navigator.platform;

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  psychoJS.setRedirectUrls('https://app.prolific.co/submissions/complete?cc=4A9C5E04', ''); //*****
  return Scheduler.Event.NEXT;
}

function downloadFiles() {
    
   // important constants
   var nBlocks = 5;
   var pid = expInfo["participant"];
   var n_files = 100;    //// to edit
     
   // get filenames of files to download
   var master_cond_file = [
       { name: 'cond_file_list_valid2_' + pid + '.csv', path: 'resources/cond_file_list_valid2_' + pid + '.csv' }
     ];
   var cond_files = new Array(nBlocks);
   for (var i = 0; i < nBlocks; i++) {
       cond_files[i] = { name: 'label_list_valid2_' + pid + '_' + i.toString() + '.csv', path: 'resources/label_list_valid2_' + pid + '_' + i.toString() + '.csv' };
   }
   
   // download resources ******   //// to edit
   var rsrc = master_cond_file.concat(cond_files);
   psychoJS.downloadResources(rsrc);

   return Scheduler.Event.NEXT;
    
}

var fix_start;
var fix_end;
var start_instructionsClock;
var instructions_text;
var key_resp;
var start_instructions2Clock;
var instructions_text_2;
var key_resp_9;
var query_sizeClock;
var query_size_text;
var x_size;
var key_resp_8;
var instructions2_text;
var text;
var trialClock;
var stimulus;
var pauseClock;
var pause_text;
var key_resp_7;
var endClock;
var end_text;
var typingClock;
var textX;
var textX_2;
var textXb;
var textXb_2;
var textXc;
var textXc_2;
var textX_3;
var textXb_3;
var textXc_3;
var allLetters;
var globalClock;
var routineTimer;
function experimentInit() {
    
  // Initialize components for Routine "start_instructions"
  start_instructionsClock = new util.Clock();
  instructions_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'instructions_text',
    text: "PLEASE READ THE FOLLOWING INSTRUCTIONS CAREFULLY. \n\nThe experiment should last between 25 and 35 minutes approximately. You will view pictures and type words using your keyboard. There will be 5 groups of pictures and you get a short break between groups.  \n\nYou should run the task in full-screen mode on a computer, rather than on a mobile device or tablet. Please turn off your phone (or put into 'do not disturb' mode) and stop any background music/videos. Do your best to find a place to do the experiment where you won't be interrupted or distracted. Please sit one arm's length (with fingers extended) away from your screen. \n\nPress the SPACE button to continue.",
    font: 'Arial',
    units: 'height',
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "start_instructions2"
  start_instructions2Clock = new util.Clock();
  instructions_text_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'instructions_text_2',
    text: "We will show you pictures that depict an object or other visual concept than can be described with only one word (e.g., rain, or cup). The object/concept depicted may be unclear. It may occupy the whole image or only a part of the image. It may be depicted many times across the image. We will ask you to write the object or concept you think the image depicts. We ask you to write three guesses, in order, from the most likely to the least likely. \n\nWe will show each picture for 2 seconds. Then, it will disappear and you will be able to type your first guess. Press ENTER to go to the next guess. When you press ENTER after entering your third guess, the experiment will go to the next picture. Please restrict each response to one word. Note that each image is associated to a different word, but that singular/plural forms and synonyms (e.g., human, person, people) all count as different words. Please answer quickly. \n\nPress SPACE to continue.",
    font: 'Arial',
    units: 'height',
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_9 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "query_size"
  query_sizeClock = new util.Clock();
  query_size_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'query_size_text',
    text: 'It is important that each participant in the experiment sees pictures of the same size. To help us show you the right size, please match the longer edge of a credit/debit card to the line below. Place your card on the screen at the start of the line and click where the card ends. Press SPACE to start the experiment.',
    font: 'Arial',
    units: 'height',
    pos: [0, 0.1], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  x_size = new visual.Slider({
    win: psychoJS.window, name: 'x_size',
    size: [1.3, 0.1], pos: [0, (- 0.4)], units: 'height',
    labels: undefined, ticks: [1, 5],
    granularity: 0, style: [visual.Slider.Style.TRIANGLE_MARKER],
    color: new util.Color('LightGray'), 
    fontFamily: 'HelveticaBold', bold: true, italic: false, 
    flip: false,
  });
  
  key_resp_8 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  stimulus = new visual.ImageStim({
    win : psychoJS.window,
    name : 'stimulus', units : 'height',
    image : undefined, mask : undefined,
    ori : 0, pos : [0, 0], size : 1.0,
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 256, interpolate : true, depth : -1.0
  });
      
    typingClock = new util.Clock();
          
    textX = new visual.TextStim({
      win: psychoJS.window,
      name: 'textX',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [0, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('black'),  opacity: 1,
      depth: -2.0
    });
    
    textX_2 = new visual.TextStim({
      win: psychoJS.window,
      name: 'textX_2',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [0, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('DarkGray'),  opacity: 1,
      depth: -4.0
    });
    
    textXb = new visual.TextStim({
      win: psychoJS.window,
      name: 'textXb',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [0, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('black'),  opacity: 1,
      depth: -2.0
    });
    
    textXb_2 = new visual.TextStim({
      win: psychoJS.window,
      name: 'textXb_2',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [0, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('DarkGray'),  opacity: 1,
      depth: -4.0
    });
    
    textXc = new visual.TextStim({
      win: psychoJS.window,
      name: 'textXc',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [0, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('black'),  opacity: 1,
      depth: -2.0
    });
    
    textXc_2 = new visual.TextStim({
      win: psychoJS.window,
      name: 'textXc_2',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [0, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('DarkGray'),  opacity: 1,
      depth: -4.0
    });
    
    textX_3 = new visual.TextStim({
      win: psychoJS.window,
      name: 'textX_3',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [-35, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('black'),  opacity: 1,
      depth: -4.0
    });
    
    textXb_3 = new visual.TextStim({
      win: psychoJS.window,
      name: 'textXb_3',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [-35, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('black'),  opacity: 1,
      depth: -4.0
    });
    
    textXc_3 = new visual.TextStim({
      win: psychoJS.window,
      name: 'textXc_3',
      text: '',
      font: 'Arial',
      units: undefined,
      pos: [-35, 0], height: 60,  wrapWidth: 1600, ori: 0,
      color: new util.Color('black'),  opacity: 1,
      depth: -4.0
    });
  
  // Initialize components for Routine "pause"
  pauseClock = new util.Clock();
  pause_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'pause_text',
    text: 'default text',
    font: 'Arial',
    units: 'height',
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  key_resp_7 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "end"
  endClock = new util.Clock();
  end_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'end_text',
    text: 'Congratulations, you are done! \nThank you for your participation! \nThe screen will close automatically.',
    font: 'Arial',
    units: 'height',
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}

var t;
var frameN;
var _key_resp_allKeys;
var start_instructionsComponents;
function start_instructionsRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'start_instructions'-------
    t = 0;
    start_instructionsClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    // keep track of which components have finished
    start_instructionsComponents = [];
    start_instructionsComponents.push(instructions_text);
    start_instructionsComponents.push(key_resp);
    
    for (const thisComponent of start_instructionsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


var continueRoutine;
function start_instructionsRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'start_instructions'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = start_instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instructions_text* updates
    if (t >= 0.0 && instructions_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instructions_text.tStart = t;  // (not accounting for frame time here)
      instructions_text.frameNStart = frameN;  // exact frame index
      
      instructions_text.setAutoDraw(true);
    }

    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }

    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of start_instructionsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function start_instructionsRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'start_instructions'-------
    for (const thisComponent of start_instructionsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        routineTimer.reset();
        }
    
    key_resp.stop();
    // the Routine "start_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_9_allKeys;
var start_instructions2Components;
function start_instructions2RoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'start_instructions2'-------
    t = 0;
    start_instructions2Clock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_9.keys = undefined;
    key_resp_9.rt = undefined;
    _key_resp_9_allKeys = [];
    // keep track of which components have finished
    start_instructions2Components = [];
    start_instructions2Components.push(instructions_text_2);
    start_instructions2Components.push(key_resp_9);
    
    for (const thisComponent of start_instructions2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function start_instructions2RoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'start_instructions2'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = start_instructions2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instructions_text_2* updates
    if (t >= 0.0 && instructions_text_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instructions_text_2.tStart = t;  // (not accounting for frame time here)
      instructions_text_2.frameNStart = frameN;  // exact frame index
      
      instructions_text_2.setAutoDraw(true);
    }

    
    // *key_resp_9* updates
    if (t >= 0.0 && key_resp_9.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_9.tStart = t;  // (not accounting for frame time here)
      key_resp_9.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_9.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_9.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_9.clearEvents(); });
    }

    if (key_resp_9.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_9.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_9_allKeys = _key_resp_9_allKeys.concat(theseKeys);
      if (_key_resp_9_allKeys.length > 0) {
        key_resp_9.keys = _key_resp_9_allKeys[_key_resp_9_allKeys.length - 1].name;  // just the last key pressed
        key_resp_9.rt = _key_resp_9_allKeys[_key_resp_9_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of start_instructions2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function start_instructions2RoutineEnd(trials) {
  return function () {
    //------Ending Routine 'start_instructions2'-------
    for (const thisComponent of start_instructions2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('key_resp_9.keys', key_resp_9.keys);
    if (typeof key_resp_9.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_9.rt', key_resp_9.rt);
        routineTimer.reset();
        }
    
    key_resp_9.stop();
    // the Routine "start_instructions2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var _key_resp_8_allKeys;
var query_sizeComponents;
function query_sizeRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'query_size'-------
    t = 0;
    query_sizeClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    x_size.reset()
    key_resp_8.keys = undefined;
    key_resp_8.rt = undefined;
    _key_resp_8_allKeys = [];
    // keep track of which components have finished
    query_sizeComponents = [];
    query_sizeComponents.push(query_size_text);
    query_sizeComponents.push(x_size);
    query_sizeComponents.push(key_resp_8);
    
    for (const thisComponent of query_sizeComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function query_sizeRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'query_size'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = query_sizeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *query_size_text* updates
    if (t >= 0.0 && query_size_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      query_size_text.tStart = t;  // (not accounting for frame time here)
      query_size_text.frameNStart = frameN;  // exact frame index
      
      query_size_text.setAutoDraw(true);
    }

    
    // *x_size* updates
    if (t >= 0.0 && x_size.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      x_size.tStart = t;  // (not accounting for frame time here)
      x_size.frameNStart = frameN;  // exact frame index
      
      x_size.setAutoDraw(true);
    }

    
    // *key_resp_8* updates
    if ((x_size.getRating()) && key_resp_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_8.tStart = t;  // (not accounting for frame time here)
      key_resp_8.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_8.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_8.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_8.clearEvents(); });
    }

    if (key_resp_8.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_8.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_8_allKeys = _key_resp_8_allKeys.concat(theseKeys);
      if (_key_resp_8_allKeys.length > 0) {
        key_resp_8.keys = _key_resp_8_allKeys[_key_resp_8_allKeys.length - 1].name;  // just the last key pressed
        key_resp_8.rt = _key_resp_8_allKeys[_key_resp_8_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of query_sizeComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function query_sizeRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'query_size'-------
    for (const thisComponent of query_sizeComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('x_size.response', x_size.getRating());
    psychoJS.experiment.addData('x_size.rt', x_size.getRT());
    psychoJS.experiment.addData('key_resp_8.keys', key_resp_8.keys);
    if (typeof key_resp_8.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_8.rt', key_resp_8.rt);
        routineTimer.reset();
        }
    
    key_resp_8.stop();
    // the Routine "query_size" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var blocks;
var currentLoop;
function blocksLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  blocks = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'cond_file_list_valid2_' + expInfo["participant"] + '.csv',
    seed: undefined, name: 'blocks'
  });
  psychoJS.experiment.addLoop(blocks); // add the loop to the experiment
  currentLoop = blocks;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisBlock of blocks) {
    const snapshot = blocks.getSnapshot();
    thisScheduler.add(importConditions(snapshot));
    const trialsLoopScheduler = new Scheduler(psychoJS);
    thisScheduler.add(trialsLoopBegin, trialsLoopScheduler);
    thisScheduler.add(trialsLoopScheduler);
    thisScheduler.add(trialsLoopEnd);
    thisScheduler.add(pauseRoutineBegin(snapshot));
    thisScheduler.add(pauseRoutineEachFrame(snapshot));
    thisScheduler.add(pauseRoutineEnd(snapshot));
    thisScheduler.add(endLoopIteration(thisScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


var trials;
function trialsLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL, /// we randomize ourselves
    extraInfo: expInfo, originPath: undefined,
    trialList: label_file,
    seed: undefined, name: 'trials'
  });
  psychoJS.experiment.addLoop(trials); // add the loop to the experiment
  currentLoop = trials;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial of trials) {
    const snapshot = trials.getSnapshot();
    thisScheduler.add(importConditions(snapshot));
    thisScheduler.add(trialRoutineBegin(snapshot));
    thisScheduler.add(trialRoutineEachFrame(snapshot));
    thisScheduler.add(trialRoutineEnd(snapshot));
    thisScheduler.add(typingRoutineBegin(snapshot));
    thisScheduler.add(typingRoutineEachFrame(snapshot));
    thisScheduler.add(typingRoutineEnd(snapshot));
    thisScheduler.add(endLoopIteration(thisScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trialsLoopEnd() {
  psychoJS.experiment.removeLoop(trials);

  return Scheduler.Event.NEXT;
}


function blocksLoopEnd() {
  psychoJS.experiment.removeLoop(blocks);

  return Scheduler.Event.NEXT;
}


var trials_2;
function trials_2LoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  trials_2 = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'label_list_valid2_' + expInfo["participant"] + '_4.csv', ////*****************
    seed: undefined, name: 'trials_2'
  });
  psychoJS.experiment.addLoop(trials_2); // add the loop to the experiment
  currentLoop = trials_2;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisTrial_2 of trials_2) {
    const snapshot = trials_2.getSnapshot();
    thisScheduler.add(importConditions(snapshot));
    thisScheduler.add(trialRoutineBegin(snapshot));
    thisScheduler.add(trialRoutineEachFrame(snapshot));
    thisScheduler.add(trialRoutineEnd(snapshot));
    thisScheduler.add(typingRoutineBegin(snapshot));
    thisScheduler.add(typingRoutineEachFrame(snapshot));
    thisScheduler.add(typingRoutineEnd(snapshot));
    thisScheduler.add(endLoopIteration(thisScheduler, snapshot));
  }

  return Scheduler.Event.NEXT;
}


function trials_2LoopEnd() {
  psychoJS.experiment.removeLoop(trials_2);

  return Scheduler.Event.NEXT;
}


var trialComponents;
var custom_size;
function trialRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'trial'-------
    t = 0;
    trialClock.reset(); // clock
    frameN = -1;
    routineTimer.add(2.200000);
    // update component parameters for each repeat
    custom_size = [((x_size.getRating() - 1) / 3.07)/1.29, ((x_size.getRating() - 1) / 3.07)/1.29];
                                                            
    stimulus.setSize(custom_size);
    stimulus.setImage('Images/valid2_'+im_nb.toString()+'.png');
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(stimulus);

    document.body.style.cursor='none'; //***//

    for (const thisComponent of trialComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}

var frameRemains;
function trialRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'trial'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    var blank_dur = 0.2;
    var stim_dur = 2.0;
                                                            
    // *stimulus* updates
    if (t >= blank_dur && stimulus.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      stimulus.tStart = t;  // (not accounting for frame time here)
      stimulus.frameNStart = frameN;  // exact frame index
      
      stimulus.setAutoDraw(true);
    }
                                                            
    frameRemains = blank_dur + stim_dur - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (stimulus.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      stimulus.setAutoDraw(false);
    }
                                                            
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of trialComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function trialRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'trial'-------
    for (const thisComponent of trialComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    
    return Scheduler.Event.NEXT;
  };
}


var t;
var frameN;
var trialComponents;
var textFill;
var textCont;
var textAdd;
var labNum;
var currLabel;
function typingRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'trial'-------
    t = 0;
    currLabel = 0;
    trialClock.reset(); // clock
    frameN = -1;

    textFill = "";
    textCont = "";
    textAdd = "";
    labNum = "1) ";
                                                            
    psychoJS.eventManager.clearKeys(); //////
      
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(textX);
    trialComponents.push(textX_2);
    trialComponents.push(textXb);
    trialComponents.push(textXb_2);
    trialComponents.push(textXc);
    trialComponents.push(textXc_2);
    trialComponents.push(textX_3);
    trialComponents.push(textXb_3);
    trialComponents.push(textXc_3);

    for (const thisComponent of trialComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
                                                            
    // *textX_2* updates LABEL 0
    if (t >= 0.0 && textX_3.status === PsychoJS.Status.NOT_STARTED  && currLabel===0) {
      // keep track of start time/frame for later
      textX_3.tStart = t;  // (not accounting for frame time here)
      textX_3.frameNStart = frameN;  // exact frame index
      
      textX_3.setAutoDraw(true);
    }
    if (textX_3.status === PsychoJS.Status.STARTED && currLabel===0){ // only update if being drawn
      textX_3.setPos([-35, 60]);
      textX_3.setText(labNum);
    }
    
    return Scheduler.Event.NEXT;
  };
}

var word_start;
var word_length;
function typingRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'trial'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = typingClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    let keys = psychoJS.eventManager.getKeys();
      
      if (keys.length > 0) {
        textAdd = keys[keys.length-1]; // so keys shouldnt be pressed too fast? otherwise would not work for checks afterwards...
        
        // check for quit (typically the Esc key)
        if (psychoJS.experiment.experimentEnded || textAdd === 'escape') {
          return psychoJS.quit('The [Escape] key was pressed. Goodbye!', false);
        }
          
        if (textAdd === 'return' && !(textFill.length<1)) {
                                                                                                            
            // erase textCont of previous label
           if (t >= 0.0 && textX_2.status === PsychoJS.Status.NOT_STARTED  && currLabel===0) {
             // keep track of start time/frame for later
             textX_2.tStart = t;  // (not accounting for frame time here)
             textX_2.frameNStart = frameN;  // exact frame index
             
             textX_2.setAutoDraw(true);
           }
           if (textX_2.status === PsychoJS.Status.STARTED && currLabel===0){ // only update if being drawn
             textX_2.setPos([(tw/2 + tw2/2), 60]);
             textX_2.setText("");
           }
           if (t >= 0.0 && textXb_2.status === PsychoJS.Status.NOT_STARTED  && currLabel===1) {
             // keep track of start time/frame for later
             textXb_2.tStart = t;  // (not accounting for frame time here)
             textXb_2.frameNStart = frameN;  // exact frame index
             
             textXb_2.setAutoDraw(true);
           }
           if (textXb_2.status === PsychoJS.Status.STARTED && currLabel===1){ // only update if being drawn
             textXb_2.setPos([(tw/2 + tw2/2), 0]);
             textXb_2.setText("");
           }
           if (t >= 0.0 && textXc_2.status === PsychoJS.Status.NOT_STARTED  && currLabel===2) {
             // keep track of start time/frame for later
             textXc_2.tStart = t;  // (not accounting for frame time here)
             textXc_2.frameNStart = frameN;  // exact frame index
             
             textXc_2.setAutoDraw(true);
           }
           if (textXc_2.status === PsychoJS.Status.STARTED && currLabel===2){ // only update if being drawn
             textXc_2.setPos([(tw/2 + tw2/2), -60]);
             textXc_2.setText("");
           }
                                                            
        // SAVE LABEL
        psychoJS.experiment.addData('label' + currLabel.toString(), textFill);
        //psychoJS.experiment.addData('resCont' + currLabel.toString(), textCont);
                                                            
        var words = textFill.split(' ');

                if (currLabel < 2) {
                                                            
                     currLabel += 1;
                     textFill = '';
                     labNum = (currLabel+1).toString() + ') ';
                                                            
                } else {
                                                            
                    continueRoutine = false;
                    textFill = '';
                                                            
                    // erase all text
                    if (t >= 0.0 && textX.status === PsychoJS.Status.NOT_STARTED) {
                      // keep track of start time/frame for later
                      textX.tStart = t;  // (not accounting for frame time here)
                      textX.frameNStart = frameN;  // exact frame index
                      
                      textX.setAutoDraw(true);
                    }
                    if (textX.status === PsychoJS.Status.STARTED){ // only update if being drawn
                      textX.setPos([0, 60])
                      textX.setText("");
                    }
                    if (t >= 0.0 && textXb.status === PsychoJS.Status.NOT_STARTED) {
                      // keep track of start time/frame for later
                      textXb.tStart = t;  // (not accounting for frame time here)
                      textXb.frameNStart = frameN;  // exact frame index
                      
                      textXb.setAutoDraw(true);
                    }
                    if (textXb.status === PsychoJS.Status.STARTED){ // only update if being drawn
                      textXb.setPos([0, 0])
                      textXb.setText("");
                    }
                    if (t >= 0.0 && textXc.status === PsychoJS.Status.NOT_STARTED) {
                      // keep track of start time/frame for later
                      textXc.tStart = t;  // (not accounting for frame time here)
                      textXc.frameNStart = frameN;  // exact frame index
                      
                      textXc.setAutoDraw(true);
                    }
                    if (textXc.status === PsychoJS.Status.STARTED){ // only update if being drawn
                      textXc.setPos([0, -60])
                      textXc.setText("");
                    }
                                                                                                        
                }
         } else if (textAdd === 'space' & textFill.length<23) {
             textFill += " ";  // Add a space
         } else if (textAdd === 'backspace') {
             textFill = textFill.slice(0, -1);
         } else if (textAdd.length===1 & textFill.length<23) { // just (lower case) letters or numbers will be accepted
             textFill += textAdd;
         }
      }
                                                            
    function getTextWidth(text, font) {
        // re-use canvas object for better performance
        var canvas = getTextWidth.canvas || (getTextWidth.canvas = document.createElement("canvas"));
        var context = canvas.getContext("2d");
        context.font = font;
        var metrics = context.measureText(text);
        return metrics.width;
    }

    var tw = getTextWidth(textFill.replace(/\s+$/, ''), "45pt arial");
    var tw2 = getTextWidth(textCont, "45pt arial");
    
    // *textX* updates LABEL 0
    if (keys.length>0 && t >= 0.0 && textX.status === PsychoJS.Status.NOT_STARTED && currLabel===0) {
      // keep track of start time/frame for later
      textX.tStart = t;  // (not accounting for frame time here)
      textX.frameNStart = frameN;  // exact frame index
      
      textX.setAutoDraw(true);
    }
    if (keys.length>0 && textX.status === PsychoJS.Status.STARTED && currLabel===0){ // only update if being drawn
      textX.setPos([0, 60])
      textX.setText(textFill);
    }
        
    // *textX_2* updates LABEL 0
    if (keys.length>0 && t >= 0.0 && textX_2.status === PsychoJS.Status.NOT_STARTED  && currLabel===0) {
      // keep track of start time/frame for later
      textX_2.tStart = t;  // (not accounting for frame time here)
      textX_2.frameNStart = frameN;  // exact frame index
      
      textX_2.setAutoDraw(true);
    }
    //if (keys.length>0 && textX_2.status === PsychoJS.Status.STARTED && currLabel===0){ // only update if being drawn
    //  textX_2.setPos([(tw/2 + tw2/2), 60]);
    //  textX_2.setText(textCont);
    //}
                                                            
    // *textX_2* updates LABEL 0
    if (keys.length>0 && t >= 0.0 && textX_3.status === PsychoJS.Status.NOT_STARTED  && currLabel===0) {
      // keep track of start time/frame for later
      textX_3.tStart = t;  // (not accounting for frame time here)
      textX_3.frameNStart = frameN;  // exact frame index
      
      textX_3.setAutoDraw(true);
    }
    if (keys.length>0 && textX_3.status === PsychoJS.Status.STARTED && currLabel===0){ // only update if being drawn
      textX_3.setPos([-(tw/2 + 35), 60]);
      //textX_3.setText(labNum);
    }
                                                            
    // *textX* updates LABEL 1
    if (keys.length>0 && t >= 0.0 && textXb.status === PsychoJS.Status.NOT_STARTED && currLabel===1) {
      // keep track of start time/frame for later
      textXb.tStart = t;  // (not accounting for frame time here)
      textXb.frameNStart = frameN;  // exact frame index
      
      textXb.setAutoDraw(true);
    }
    if (keys.length>0 && textXb.status === PsychoJS.Status.STARTED && currLabel===1){ // only update if being drawn
      textXb.setPos([0, 0])
      textXb.setText(textFill);
    }
        
    // *textX_2* updates LABEL 1
    if (keys.length>0 && t >= 0.0 && textXb_2.status === PsychoJS.Status.NOT_STARTED  && currLabel===1) {
      // keep track of start time/frame for later
      textXb_2.tStart = t;  // (not accounting for frame time here)
      textXb_2.frameNStart = frameN;  // exact frame index
      
      textXb_2.setAutoDraw(true);
    }
    //if (keys.length>0 && textXb_2.status === PsychoJS.Status.STARTED && currLabel===1){ // only update if being drawn
    //  textXb_2.setPos([(tw/2 + tw2/2), 0]);
    //  textXb_2.setText(textCont);
    //}
                                                            
    // *textX_2* updates LABEL 1
    if (keys.length>0 && t >= 0.0 && textXb_3.status === PsychoJS.Status.NOT_STARTED  && currLabel===1) {
      // keep track of start time/frame for later
      textXb_3.tStart = t;  // (not accounting for frame time here)
      textXb_3.frameNStart = frameN;  // exact frame index
      
      textXb_3.setAutoDraw(true);
    }
    if (keys.length>0 && textXb_3.status === PsychoJS.Status.STARTED && currLabel===1){ // only update if being drawn
      textXb_3.setPos([-(tw/2 + 35), 0]);
      textXb_3.setText(labNum);
    }
                                                            
    // *textX* updates LABEL 2
    if (keys.length>0 && t >= 0.0 && textXc.status === PsychoJS.Status.NOT_STARTED && currLabel===2) {
      // keep track of start time/frame for later
      textXc.tStart = t;  // (not accounting for frame time here)
      textXc.frameNStart = frameN;  // exact frame index
      
      textXc.setAutoDraw(true);
    }
    if (keys.length>0 && textXc.status === PsychoJS.Status.STARTED && currLabel===2){ // only update if being drawn
      textXc.setPos([0, -60])
      textXc.setText(textFill);
    }
        
    // *textX_2* updates LABEL 2
    if (keys.length>0 && t >= 0.0 && textXc_2.status === PsychoJS.Status.NOT_STARTED  && currLabel===2) {
      // keep track of start time/frame for later
      textXc_2.tStart = t;  // (not accounting for frame time here)
      textXc_2.frameNStart = frameN;  // exact frame index
      
      textXc_2.setAutoDraw(true);
    }
    //if (keys.length>0 && textXc_2.status === PsychoJS.Status.STARTED && currLabel===2){ // only update if being drawn
    //  textXc_2.setPos([(tw/2 + tw2/2), -60]);
    //  textXc_2.setText(textCont);
    //}
                                                            
    // *textX_2* updates LABEL 2
    if (keys.length>0 && t >= 0.0 && textXc_3.status === PsychoJS.Status.NOT_STARTED  && currLabel===2) {
      // keep track of start time/frame for later
      textXc_3.tStart = t;  // (not accounting for frame time here)
      textXc_3.frameNStart = frameN;  // exact frame index
      
      textXc_3.setAutoDraw(true);
    }
    if (keys.length>0 && textXc_3.status === PsychoJS.Status.STARTED && currLabel===2){ // only update if being drawn
      textXc_3.setPos([-(tw/2 + 35), -60]);
      textXc_3.setText(labNum);
    }
                                                            
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of trialComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}

function typingRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'trial'-------
    for (const thisComponent of trialComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }

    // the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}
                                                            

var _key_resp_7_allKeys;
var pauseComponents;
function pauseRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'pause'-------
    t = 0;
    pauseClock.reset(); // clock
    frameN = -1;
                 
    // update component parameters for each repeat
    var nRemBlocks = 5 - Number(label_file.slice(-5,-4)) - 1; ////// ***********
    var msg = ("You can take a few seconds before pressing SPACE to continue. \n\nNumber of remaining blocks: ").concat(nRemBlocks.toString(), "/5 ");
    pause_text.setText(msg);
                                                            
    key_resp_7.keys = undefined;
    key_resp_7.rt = undefined;
    _key_resp_7_allKeys = [];
    // keep track of which components have finished
    pauseComponents = [];
    pauseComponents.push(pause_text);
    pauseComponents.push(key_resp_7);
    
    for (const thisComponent of pauseComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function pauseRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'pause'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = pauseClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *pause_text* updates
    if (t >= 0.0 && pause_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      pause_text.tStart = t;  // (not accounting for frame time here)
      pause_text.frameNStart = frameN;  // exact frame index
      
      pause_text.setAutoDraw(true);
    }

    
    // *key_resp_7* updates
    if (t >= 0.0 && key_resp_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_7.tStart = t;  // (not accounting for frame time here)
      key_resp_7.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_7.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_7.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_7.clearEvents(); });
    }

    if (key_resp_7.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_7.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_7_allKeys = _key_resp_7_allKeys.concat(theseKeys);
      if (_key_resp_7_allKeys.length > 0) {
        key_resp_7.keys = _key_resp_7_allKeys[_key_resp_7_allKeys.length - 1].name;  // just the last key pressed
        key_resp_7.rt = _key_resp_7_allKeys[_key_resp_7_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of pauseComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function pauseRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'pause'-------
    for (const thisComponent of pauseComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('key_resp_7.keys', key_resp_7.keys);
    if (typeof key_resp_7.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_7.rt', key_resp_7.rt);
        routineTimer.reset();
        }
    
    key_resp_7.stop();
    // the Routine "pause" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var endComponents;
function endRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'end'-------
    t = 0;
    endClock.reset(); // clock
    frameN = -1;
    routineTimer.add(5.000000);
    // update component parameters for each repeat
    // keep track of which components have finished
    endComponents = [];
    endComponents.push(end_text);
                                                            
    document.body.style.cursor='auto'; //***//
    
    for (const thisComponent of endComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}

var frameRemains;
function endRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'end'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = endClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *end_text* updates
    if (t >= 0.0 && end_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      end_text.tStart = t;  // (not accounting for frame time here)
      end_text.frameNStart = frameN;  // exact frame index
      
      end_text.setAutoDraw(true);
    }

    frameRemains = 0.0 + 5 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (end_text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      end_text.setAutoDraw(false);
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of endComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function endRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'end'-------
    for (const thisComponent of endComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    return Scheduler.Event.NEXT;
  };
}


function endLoopIteration(thisScheduler, loop) {
  // ------Prepare for next entry------
  return function () {
    if (typeof loop !== 'undefined') {
      // ------Check if user ended loop early------
      if (loop.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(loop);
        }
      thisScheduler.stop();
      } else {
        const thisTrial = loop.getCurrentTrial();
        if (typeof thisTrial === 'undefined' || !('isTrials' in thisTrial) || thisTrial.isTrials) {
          psychoJS.experiment.nextEntry(loop);
        }
      }
    return Scheduler.Event.NEXT;
    }
  };
}


function importConditions(trials) {
  return function () {
    psychoJS.importAttributes(trials.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
