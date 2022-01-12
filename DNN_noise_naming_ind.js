/**************************
 * Dnn_noise_naming Experiment *

 JS code created with PsychoPy Builder and edited manually
 Copyright (c) Laurent Caplette 2021
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
let expName = 'DNN_noise_naming_ind';  // from the Builder filename that created this script
let expInfo = {'participant': '', 'session': ''};

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
flowScheduler.add(practice_instructionsRoutineBegin());
flowScheduler.add(practice_instructionsRoutineEachFrame());
flowScheduler.add(practice_instructionsRoutineEnd());
const practiceLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(practiceLoopBegin, practiceLoopScheduler);
flowScheduler.add(practiceLoopScheduler);
flowScheduler.add(practiceLoopEnd);
flowScheduler.add(main_instructionsRoutineBegin());
flowScheduler.add(main_instructionsRoutineEachFrame());
flowScheduler.add(main_instructionsRoutineEnd());
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

var n_files_prac = 5;
var prac_im_files = new Array(n_files_prac);
for (var i = 0; i < n_files_prac; i++) {
  prac_im_files[i] = { name: 'Images/practice/prac_im_' + i.toString() + '.png', path: 'resources/Images/practice/prac_im_' + i.toString() + '.png' };
}
var prac_im_lists = [{ name: 'im_list_practice_even.csv', path: 'resources/im_list_practice_even.csv' }];
var dict = [{name: 'concreteness.csv', path: 'resources/concreteness.csv'}];
var start_files = prac_im_lists.concat(prac_im_files).concat(dict);

psychoJS.start({
expName: expName,
expInfo: expInfo,
resources: start_files
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
  //psychoJS.setRedirectUrls('address', ''); //***** Put SONA address in first field
  return Scheduler.Event.NEXT;
}

function downloadFiles() {
    
   // important constants
   var nBlocks = 5;
   var pid = expInfo["participant"]; // 0-7 (only stimuli order changes)
   var sid = expInfo["session"]; // 0-5 (stimuli change)
   var n_files = 125;
     
   // get filenames of files to download
   var master_cond_file = [
       { name: 'cond_file_list_2C_' + pid + '_' + sid + '.csv', path: 'resources/cond_file_list_2C_' + pid + '_' + sid + '.csv' }
     ];
   var cond_files = new Array(nBlocks);
   for (var i = 0; i < nBlocks; i++) {
       cond_files[i] = { name: 'im_list_2C_' + pid + '_' + sid + '_' + i.toString() + '.csv', path: 'resources/im_list_2C_' + pid + '_' + sid + '_' + i.toString() + '.csv' };
   }
   var im_files = new Array(n_files);
   for (var i = 0; i < n_files; i++) {
       im_files[i] = { name: 'Images/2C_' + pid + '_' + sid + '/stim_2C_' + pid + '_' + sid + '_' + i.toString() + '.png', path: 'resources/Images/2C_' + pid + '_' + sid + '/stim_2C_' + pid + '_' + sid + '_' + i.toString() + '.png' };
   }
   
   // download resources ******
   var rsrc = master_cond_file.concat(cond_files);
   psychoJS.downloadResources(rsrc);
   psychoJS.downloadResources(im_files.slice(0,43)); // seems we need to do this to prevent weird bug...
   psychoJS.downloadResources(im_files.slice(40,83));
   psychoJS.downloadResources(im_files.slice(80,125));
   psychoJS.downloadResources(im_files.slice(124));

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
var practice_instructionsClock;
var instructions2_text;
var key_resp_3;
var text;
var main_instructionsClock;
var main_instructions_text;
var key_resp_4;
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
var predictionary;
var globalClock;
var routineTimer;
function experimentInit() {
    
  // Initialize components for Routine "start_instructions"
  start_instructionsClock = new util.Clock();
  instructions_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'instructions_text',
    text: "PLEASE READ THE FOLLOWING INSTRUCTIONS CAREFULLY. \n\nThe experiment should last between 30 and 40 minutes approximately. You will view pictures and type words using your keyboard. There will be 5 groups of pictures and you get a short break between groups.  \n\nYou should run the task in full-screen mode on a computer, rather than on a mobile device or tablet. Please turn off your phone (or put into 'do not disturb' mode) and stop any background music/videos. Do your best to find a place to do the experiment where you won't be interrupted or distracted. Please sit one arm's length (with fingers extended) away from your screen. \n\nPress the SPACE button to continue.",
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
    text: "PLEASE READ THE FOLLOWING INSTRUCTIONS CAREFULLY. \n\nWe will show you dream-like pictures. Unclear or distorted objects, object parts and textures may be seen in the pictures. We will ask you to indicate what objects or concrete things you see (you might need to use your imagination!). You need to indicate at least one thing per picture and at most three. Try to indicate three when you can. \n\nWe will show each picture for 5 seconds. Then, it will disappear and you will be able to type the first thing you saw. Please be concise and try to write only ONE NOUN per thing as much as possible (e.g. 'cat' and not 'black cat' or 'big scary cat'). Please answer quickly. To increase the speed, words will be automatically suggested based on the letters you type. You can ignore them or press TAB to accept the suggestions. Press ENTER to go to the next label. You can leave the second and/or third label empty if you did not see more than one thing. When you press ENTER at the third label, the experiment will go to the next picture. \n\nPress SPACE to continue.",
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
    text: 'It is important that each participant in the experiment sees pictures of the same size. To help us show you the right size, please match the longer edge of a credit/debit card to the line below. Place your card on the screen at the start of the line and click where the card ends. Press SPACE when you are done.',
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
  
  // Initialize components for Routine "practice_instructions"
  practice_instructionsClock = new util.Clock();
  instructions2_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'instructions2_text',
    text: "Before we start the real experiment, we will do a brief round of practice. We will show you 5 pictures. After each picture, you'll be able to type between 1 and 3 labels to indicate what you saw in the picture. Press TAB to accept a word suggestion, press ENTER to go to the next label or, if you're at the third label, end the trial and go to the next picture. Try to not get mixed between TAB and ENTER if you use both. You can use this practice round to practice using TAB. \n\nGood luck!\nPress SPACE to start.",
    font: 'Arial',
    units: 'height',
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "main_instructions"
  main_instructionsClock = new util.Clock();
  main_instructions_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'main_instructions_text',
    text: "We will now start the main experiment. There will be 5 groups of pictures with short breaks in-between. Please eliminate all distractions and do your best to concentrate on the task. Please sit one arm's length (with fingers extended) away from your screen. \n\nREMINDER: press TAB to accept a word suggestion, press ENTER to go to the next label or go to the next picture if you're at the third label. Please try to answer quickly and concisely! \n\nGood luck!\nPress SPACE to start",
    font: 'Arial',
    units: 'height',
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0,
    color: new util.Color('black'),  opacity: 1,
    depth: 0.0 
  });
  
  key_resp_4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
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
    
    predictionary = Predictionary.instance();
      $.get('resources/concreteness.csv', function (result) {
          parseWords(result);
      });

      function parseWords(string) {
          predictionary.parseWords(string, {
              elementSeparator: '\n',
              rankSeparator: ';',
              wordPosition: 0,
              rankPosition: 5
          });
      }
          
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


var _key_resp_3_allKeys;
var practice_instructionsComponents;
function practice_instructionsRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'practice_instructions'-------
    t = 0;
    practice_instructionsClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_3.keys = undefined;
    key_resp_3.rt = undefined;
    _key_resp_3_allKeys = [];
    // keep track of which components have finished
    practice_instructionsComponents = [];
    practice_instructionsComponents.push(instructions2_text);
    practice_instructionsComponents.push(key_resp_3);
          
    for (const thisComponent of practice_instructionsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function practice_instructionsRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'practice_instructions'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = practice_instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instructions2_text* updates
    if (t >= 0.0 && instructions2_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instructions2_text.tStart = t;  // (not accounting for frame time here)
      instructions2_text.frameNStart = frameN;  // exact frame index
      
      instructions2_text.setAutoDraw(true);
    }
      
      document.body.style.cursor='none'; //***//
    
    // *key_resp_3* updates
    if (t >= 0.0 && key_resp_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_3.tStart = t;  // (not accounting for frame time here)
      key_resp_3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_3.clearEvents(); });
    }

    if (key_resp_3.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_3.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_3_allKeys = _key_resp_3_allKeys.concat(theseKeys);
      if (_key_resp_3_allKeys.length > 0) {
        key_resp_3.keys = _key_resp_3_allKeys[_key_resp_3_allKeys.length - 1].name;  // just the last key pressed
        key_resp_3.rt = _key_resp_3_allKeys[_key_resp_3_allKeys.length - 1].rt;
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
    for (const thisComponent of practice_instructionsComponents)
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


function practice_instructionsRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'practice_instructions'-------
    for (const thisComponent of practice_instructionsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('key_resp_3.keys', key_resp_3.keys);
    if (typeof key_resp_3.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_3.rt', key_resp_3.rt);
        routineTimer.reset();
        }
    
    key_resp_3.stop();
    // the Routine "practice_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}

var practice;
var currentLoop;
function practiceLoopBegin(thisScheduler) {
      
    var practicefile = 'im_list_practice_even.csv';

  practice = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: practicefile,
    seed: undefined, name: 'practice'
  });
  psychoJS.experiment.addLoop(practice); // add the loop to the experiment
  currentLoop = practice;  // we're now the current loop

  // Schedule all the trials in the trialList:
  for (const thisPractice of practice) {
    const snapshot = practice.getSnapshot();
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

function practiceLoopEnd() {
  psychoJS.experiment.removeLoop(practice);

  return Scheduler.Event.NEXT;
}

var blocks;
function blocksLoopBegin(thisScheduler) {
  // set up handler to look after randomisation of conditions etc
  blocks = new TrialHandler({
    psychoJS: psychoJS,
    nReps: 1, method: TrialHandler.Method.SEQUENTIAL,
    extraInfo: expInfo, originPath: undefined,
    trialList: 'cond_file_list_2C_' + expInfo["participant"] + '_' + expInfo["session"] + '.csv',
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
    trialList: im_file,
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
    trialList: 'im_list_2C_' + expInfo["participant"] + '_' + expInfo["session"] + '_4.csv', ////*****************
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
                                                            
var _key_resp_4_allKeys;
var main_instructionsComponents;
function main_instructionsRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'main_instructions'-------
    t = 0;
    main_instructionsClock.reset(); // clock
    frameN = -1;
    // update component parameters for each repeat
    key_resp_4.keys = undefined;
    key_resp_4.rt = undefined;
    _key_resp_4_allKeys = [];
    // keep track of which components have finished
    main_instructionsComponents = [];
    main_instructionsComponents.push(main_instructions_text);
    main_instructionsComponents.push(key_resp_4);
    
    for (const thisComponent of main_instructionsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    
    return Scheduler.Event.NEXT;
  };
}


function main_instructionsRoutineEachFrame(trials) {
  return function () {
    //------Loop for each frame of Routine 'main_instructions'-------
    let continueRoutine = true; // until we're told otherwise
    // get current time
    t = main_instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *main_instructions_text* updates
    if (t >= 0.0 && main_instructions_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      main_instructions_text.tStart = t;  // (not accounting for frame time here)
      main_instructions_text.frameNStart = frameN;  // exact frame index
      
      main_instructions_text.setAutoDraw(true);
    }

    
    // *key_resp_4* updates
    if (t >= 0.0 && key_resp_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_4.tStart = t;  // (not accounting for frame time here)
      key_resp_4.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_4.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_4.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_4.clearEvents(); });
    }

    if (key_resp_4.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_4.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_4_allKeys = _key_resp_4_allKeys.concat(theseKeys);
      if (_key_resp_4_allKeys.length > 0) {
        key_resp_4.keys = _key_resp_4_allKeys[_key_resp_4_allKeys.length - 1].name;  // just the last key pressed
        key_resp_4.rt = _key_resp_4_allKeys[_key_resp_4_allKeys.length - 1].rt;
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
    for (const thisComponent of main_instructionsComponents)
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


function main_instructionsRoutineEnd(trials) {
  return function () {
    //------Ending Routine 'main_instructions'-------
    for (const thisComponent of main_instructionsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('key_resp_4.keys', key_resp_4.keys);
    if (typeof key_resp_4.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_4.rt', key_resp_4.rt);
        routineTimer.reset();
        }
    
    key_resp_4.stop();
    // the Routine "main_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    return Scheduler.Event.NEXT;
  };
}


var trialComponents;
var custom_size;
function trialRoutineBegin(trials) {
  return function () {
    //------Prepare to start Routine 'trial'-------
    t = 0;
    trialClock.reset(); // clock
    frameN = -1;
    routineTimer.add(5.200000);
    // update component parameters for each repeat
    custom_size = [((x_size.getRating() - 1) / 3.07)/1.29, ((x_size.getRating() - 1) / 3.07)/1.29];
                                                            
    stimulus.setSize(custom_size);
    stimulus.setImage(im_path);
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(stimulus);

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
    var stim_dur = 5.0;
                                                            
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
          
        if (textAdd === 'return' && !(currLabel===0 && textFill.length<1)) {
                                                                                                            
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
        psychoJS.experiment.addData('resCont' + currLabel.toString(), textCont);
                                                            
        // learn from written words
        var words = textFill.split(' ');
        for (var i = 0; i < words.length; i++) {
            predictionary.learn(words[i]);
        }

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
         } else if (textAdd === 'tab' & textCont.length > 0) {
             textFill += textCont;
             textFill += " ";
         } else if (textAdd.length===1 & textFill.length<23) { // just (lower case) letters or numbers will be accepted
             textFill += textAdd;
         }
      }
            
    if (keys.length > 0 & textAdd.length > 0 & textFill[-1]!==' ') {  // at least one letter/number was pressed
        word_start = (textFill.lastIndexOf(" ") + 1);
        word_length = textFill.length - word_start;
        ////// maybe run predict only if new keys are incompatible with last suggestion?
        let suggestions = predictionary.predict(textFill.slice(word_start));
          if (suggestions.length>0 & textFill.slice(-1)!== ' ' & textFill.length>0){
                            
              if (suggestions[0].slice(0,word_length)===textFill.slice(word_start)) {
                  textCont = suggestions[0].slice(word_length);
              } else {
                      textCont = "";
              }
          } else {
              textCont = "";
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
    if (keys.length>0 && textX_2.status === PsychoJS.Status.STARTED && currLabel===0){ // only update if being drawn
      textX_2.setPos([(tw/2 + tw2/2), 60]);
      textX_2.setText(textCont);
    }
                                                            
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
    if (keys.length>0 && textXb_2.status === PsychoJS.Status.STARTED && currLabel===1){ // only update if being drawn
      textXb_2.setPos([(tw/2 + tw2/2), 0]);
      textXb_2.setText(textCont);
    }
                                                            
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
    if (keys.length>0 && textXc_2.status === PsychoJS.Status.STARTED && currLabel===2){ // only update if being drawn
      textXc_2.setPos([(tw/2 + tw2/2), -60]);
      textXc_2.setText(textCont);
    }
                                                            
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
    var nRemBlocks = 5 - Number(im_file.slice(-5,-4)) - 1; ////// ***********
    var msg = ("You can take a few seconds before pressing SPACE to continue. \n\nREMINDER: TAB to accept suggestion, ENTER to go to next label or next trial. \n\nNumber of remaining blocks: ").concat(nRemBlocks.toString(), "/5 ");
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
