function BackendGameClient(InputManager, Actuator) {
  this.inputManager = new InputManager();
  this.actuator = new Actuator();

  this.inFlight = false;
  this.pendingDirection = null;
  this.trainingPollTimer = null;
  this.trainingElements = null;
  this.isTrainingRunning = false;
  this.trainingBestScore = 0;
  this.lastTrainingRawBoard = null;
  this.trainingAnimationMs = 140;
  this.trainingAckTimer = null;
  this.lastRenderedFrameId = 0;
  this.lastAckedFrameId = 0;
  this.trainingDefaultsApplied = false;

  this.inputManager.on("move", this.move.bind(this));
  this.inputManager.on("restart", this.restart.bind(this));
  this.inputManager.on("keepPlaying", this.keepPlaying.bind(this));

  this.setupTrainingUI();
  this.setupReplayUI();
  this.fetchState();
  this.fetchTrainingStatus();
}

BackendGameClient.prototype.fetchState = function () {
  var self = this;
  this.sendRequest("GET", "/api/state", null, function (error, state) {
    if (error) {
      self.logError(error);
      return;
    }
    self.render(state);
  });
};

BackendGameClient.prototype.sendRequest = function (method, path, data, callback) {
  var xhr = new XMLHttpRequest();

  xhr.open(method, path, true);
  xhr.setRequestHeader("Accept", "application/json");

  xhr.onreadystatechange = function () {
    if (xhr.readyState !== 4) {
      return;
    }

    if (xhr.status >= 200 && xhr.status < 300) {
      var parsed;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch (error) {
        callback(error);
        return;
      }
      callback(null, parsed);
      return;
    }

    callback(new Error("Request failed: " + method + " " + path + " (" + xhr.status + ")"));
  };

  xhr.onerror = function () {
    callback(new Error("Network error: " + method + " " + path));
  };

  if (data) {
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.send(JSON.stringify(data));
  } else {
    xhr.send();
  }
};

BackendGameClient.prototype.buildGrid = function (gridState) {
  var grid = new Grid(gridState.size, gridState.cells);

  // Mark tiles as "already positioned" so they don't all animate as new every update.
  grid.eachCell(function (x, y, tile) {
    if (tile) {
      tile.previousPosition = { x: tile.x, y: tile.y };
    }
  });

  return grid;
};

BackendGameClient.prototype.render = function (state) {
  if (!state || !state.grid) {
    return;
  }
  if (this.isTrainingRunning) {
    return;
  }

  if (!state.terminated) {
    this.actuator.continueGame();
  }

  var grid = state.animationGrid
    ? this.buildGridFromAnimationGrid(state.animationGrid)
    : this.buildGrid(state.grid);

  this.actuator.actuate(grid, {
    score: state.score || 0,
    over: !!state.over,
    won: !!state.won,
    bestScore: state.bestScore || 0,
    terminated: !!state.terminated
  });
};

BackendGameClient.prototype.restart = function () {
  if (this.isTrainingRunning) {
    return;
  }
  var self = this;
  this.pendingDirection = null;
  this.sendRequest("POST", "/api/restart", null, function (error, state) {
    if (error) {
      self.logError(error);
      return;
    }
    self.actuator.continueGame();
    self.render(state);
  });
};

BackendGameClient.prototype.keepPlaying = function () {
  if (this.isTrainingRunning) {
    return;
  }
  var self = this;
  this.sendRequest("POST", "/api/keep-playing", null, function (error, state) {
    if (error) {
      self.logError(error);
      return;
    }
    self.actuator.continueGame();
    self.render(state);
  });
};

BackendGameClient.prototype.move = function (direction) {
  if (this.isTrainingRunning) {
    return;
  }
  var self = this;

  if (this.inFlight) {
    this.pendingDirection = direction;
    return;
  }

  this.inFlight = true;
  this.sendRequest("POST", "/api/move", { direction: direction }, function (error, state) {
    self.inFlight = false;

    if (error) {
      self.logError(error);
    } else {
      self.render(state);
    }

    if (self.pendingDirection !== null) {
      var queuedDirection = self.pendingDirection;
      self.pendingDirection = null;
      self.move(queuedDirection);
    }
  });
};

BackendGameClient.prototype.setupTrainingUI = function () {
  var startButton = document.getElementById("training-start-button");
  var stopButton = document.getElementById("training-stop-button");
  var statusBlock = document.getElementById("training-status");
  var boardBlock = document.getElementById("training-board");
  var episodesInput = document.getElementById("training-episodes");
  var workersInput = document.getElementById("training-workers");
  var seedInput = document.getElementById("training-seed");
  var maxStepsInput = document.getElementById("training-max-steps");
  var checkpointEveryInput = document.getElementById("training-checkpoint-every");
  var checkpointDirInput = document.getElementById("training-checkpoint-dir");
  var loadModelPathInput = document.getElementById("training-load-model-path");
  var terminateCheckbox = document.getElementById("training-terminate-on-win");
  var playOnlyCheckbox = document.getElementById("training-play-only");

  if (!startButton || !stopButton || !statusBlock || !boardBlock) {
    return;
  }

  this.trainingElements = {
    startButton: startButton,
    stopButton: stopButton,
    statusBlock: statusBlock,
    boardBlock: boardBlock,
    episodesInput: episodesInput,
    workersInput: workersInput,
    seedInput: seedInput,
    maxStepsInput: maxStepsInput,
    checkpointEveryInput: checkpointEveryInput,
    checkpointDirInput: checkpointDirInput,
    loadModelPathInput: loadModelPathInput,
    terminateCheckbox: terminateCheckbox,
    playOnlyCheckbox: playOnlyCheckbox
  };

  var self = this;
  startButton.addEventListener("click", function () {
    self.startTraining();
  });
  stopButton.addEventListener("click", function () {
    self.stopTraining();
  });
};

BackendGameClient.prototype.getTrainingPayload = function () {
  if (!this.trainingElements) {
    return null;
  }

  var episodes = parseInt(this.trainingElements.episodesInput.value || "100", 10);
  var workers = parseInt(this.trainingElements.workersInput.value || "1", 10);
  var seedText = this.trainingElements.seedInput.value;
  var maxStepsText = this.trainingElements.maxStepsInput.value;
  var checkpointEvery = parseInt(
    (this.trainingElements.checkpointEveryInput && this.trainingElements.checkpointEveryInput.value) || "0",
    10
  );
  var checkpointDirText =
    (this.trainingElements.checkpointDirInput && this.trainingElements.checkpointDirInput.value) || "";
  var loadModelPathText =
    (this.trainingElements.loadModelPathInput && this.trainingElements.loadModelPathInput.value) || "";

  var payload = {
    episodes: isNaN(episodes) ? 100 : episodes,
    workers: isNaN(workers) ? 1 : workers,
    seed: seedText === "" ? null : parseInt(seedText, 10),
    maxSteps: maxStepsText === "" ? null : parseInt(maxStepsText, 10),
    terminateOnWin: this.trainingElements.terminateCheckbox.checked,
    // Frontend mode should block each step until animation ACK.
    syncWithFrontend: true,
    checkpointEveryEpisodes: isNaN(checkpointEvery) ? 0 : checkpointEvery,
    checkpointDir: checkpointDirText === "" ? null : checkpointDirText,
    loadModelPath: loadModelPathText === "" ? null : loadModelPathText,
    playOnly: !!(this.trainingElements.playOnlyCheckbox && this.trainingElements.playOnlyCheckbox.checked)
  };

  if (isNaN(payload.seed)) {
    payload.seed = null;
  }
  if (isNaN(payload.maxSteps)) {
    payload.maxSteps = null;
  }

  return payload;
};

BackendGameClient.prototype.startTraining = function () {
  var self = this;
  var payload = this.getTrainingPayload();
  if (!payload) {
    return;
  }

  this.sendRequest("POST", "/api/train/start", payload, function (error, status) {
    if (error) {
      self.logError(error);
      self.renderTrainingError("Failed to start training.");
      return;
    }
    self.trainingBestScore = 0;
    self.lastTrainingRawBoard = null;
    self.lastRenderedFrameId = 0;
    self.lastAckedFrameId = 0;
    if (self.trainingAckTimer) {
      window.clearTimeout(self.trainingAckTimer);
      self.trainingAckTimer = null;
    }
    self.updateTrainingStatus(status);
    self.startTrainingPolling();
  });
};

BackendGameClient.prototype.stopTraining = function () {
  var self = this;
  this.sendRequest("POST", "/api/train/stop", null, function (error, status) {
    if (error) {
      self.logError(error);
      self.renderTrainingError("Failed to stop training.");
      return;
    }
    self.updateTrainingStatus(status);
  });
};

BackendGameClient.prototype.fetchTrainingStatus = function () {
  var self = this;
  if (!this.trainingElements) {
    return;
  }

  this.sendRequest("GET", "/api/train/status", null, function (error, status) {
    if (error) {
      self.logError(error);
      self.renderTrainingError("Cannot fetch training status.");
      return;
    }
    self.updateTrainingStatus(status);
  });
};

BackendGameClient.prototype.startTrainingPolling = function () {
  var self = this;
  if (this.trainingPollTimer) {
    return;
  }
  this.trainingPollTimer = window.setInterval(function () {
    self.fetchTrainingStatus();
  }, 40);
};

BackendGameClient.prototype.stopTrainingPolling = function () {
  if (this.trainingPollTimer) {
    window.clearInterval(this.trainingPollTimer);
    this.trainingPollTimer = null;
  }
  if (this.trainingAckTimer) {
    window.clearTimeout(this.trainingAckTimer);
    this.trainingAckTimer = null;
  }
};

BackendGameClient.prototype.renderTrainingError = function (message) {
  if (!this.trainingElements) {
    return;
  }
  this.trainingElements.statusBlock.textContent = message;
};

BackendGameClient.prototype.formatTrainingBoard = function (rawBoard) {
  if (!rawBoard || !rawBoard.length) {
    return "";
  }

  var maxValue = 0;
  rawBoard.forEach(function (row) {
    row.forEach(function (value) {
      if (value > maxValue) {
        maxValue = value;
      }
    });
  });

  var width = Math.max(String(maxValue || 0).length, 2);
  return rawBoard
    .map(function (row) {
      return row
        .map(function (value) {
          var cell = value === 0 ? "." : String(value);
          while (cell.length < width) {
            cell = " " + cell;
          }
          return cell;
        })
        .join(" ");
    })
    .join("\n");
};

BackendGameClient.prototype.buildGridFromRawBoard = function (rawBoard) {
  return this.buildGridFromRawBoards(rawBoard, null);
};

BackendGameClient.prototype.buildTileFromAnimationData = function (tileData) {
  if (!tileData || !tileData.position) {
    return null;
  }

  var tile = new Tile(tileData.position, tileData.value);
  if (tileData.previousPosition) {
    tile.previousPosition = {
      x: tileData.previousPosition.x,
      y: tileData.previousPosition.y
    };
  }

  if (tileData.mergedFrom && tileData.mergedFrom.length) {
    var self = this;
    tile.mergedFrom = tileData.mergedFrom
      .map(function (mergedTileData) {
        return self.buildTileFromAnimationData(mergedTileData);
      })
      .filter(function (mergedTile) {
        return !!mergedTile;
      });
  }

  return tile;
};

BackendGameClient.prototype.buildGridFromAnimationGrid = function (animationGrid) {
  if (!animationGrid || !animationGrid.cells || !animationGrid.size) {
    return null;
  }

  var size = animationGrid.size;
  var grid = new Grid(size);
  var x;
  var y;

  for (x = 0; x < size; x++) {
    var column = animationGrid.cells[x] || [];
    for (y = 0; y < size; y++) {
      var tileData = column[y];
      grid.cells[x][y] = tileData ? this.buildTileFromAnimationData(tileData) : null;
    }
  }

  return grid;
};

BackendGameClient.prototype.buildGridFromRawBoards = function (rawBoard, previousRawBoard) {
  var size = rawBoard.length;
  var cells = [];
  var x;
  var y;

  for (x = 0; x < size; x++) {
    cells[x] = [];
    for (y = 0; y < size; y++) {
      var row = rawBoard[y] || [];
      var value = row[x] || 0;
      if (value > 0) {
        cells[x][y] = {
          position: { x: x, y: y },
          value: value
        };
      } else {
        cells[x][y] = null;
      }
    }
  }

  return this.buildGrid({ size: size, cells: cells });
};

BackendGameClient.prototype.renderTrainingOnMainBoard = function (latestState) {
  if (!latestState || !latestState.rawBoard) {
    return;
  }

  var score = latestState.score || 0;
  if (score > this.trainingBestScore) {
    this.trainingBestScore = score;
  }

  var grid = null;
  if (latestState.animationGrid) {
    grid = this.buildGridFromAnimationGrid(latestState.animationGrid);
  }
  if (!grid) {
    grid = this.buildGridFromRawBoards(latestState.rawBoard, this.lastTrainingRawBoard);
  }
  this.actuator.continueGame();
  this.actuator.actuate(grid, {
    score: score,
    over: false,
    won: false,
    bestScore: this.trainingBestScore,
    terminated: false
  });

  this.lastTrainingRawBoard = JSON.parse(JSON.stringify(latestState.rawBoard));
};

BackendGameClient.prototype.sendTrainingStepDone = function (frameId) {
  var self = this;
  this.sendRequest("POST", "/api/train/step-done", { frameId: frameId }, function (error) {
    if (error) {
      self.logError(error);
      return;
    }
    self.lastAckedFrameId = frameId;
  });
};

BackendGameClient.prototype.scheduleTrainingStepDone = function (frameId) {
  var self = this;
  if (this.trainingAckTimer) {
    window.clearTimeout(this.trainingAckTimer);
    this.trainingAckTimer = null;
  }
  this.trainingAckTimer = window.setTimeout(function () {
    self.trainingAckTimer = null;
    self.sendTrainingStepDone(frameId);
  }, this.trainingAnimationMs);
};

BackendGameClient.prototype.updateTrainingStatus = function (status) {
  if (!this.trainingElements || !status) {
    return;
  }

  if (!this.trainingDefaultsApplied && status.trainingDefaults) {
    var defaults = status.trainingDefaults;
    if (this.trainingElements.episodesInput && defaults.episodes != null) {
      this.trainingElements.episodesInput.value = String(defaults.episodes);
    }
    if (this.trainingElements.workersInput && defaults.workers != null) {
      this.trainingElements.workersInput.value = String(defaults.workers);
    }
    if (this.trainingElements.seedInput) {
      this.trainingElements.seedInput.value = defaults.seed == null ? "" : String(defaults.seed);
    }
    if (this.trainingElements.maxStepsInput) {
      this.trainingElements.maxStepsInput.value = defaults.maxSteps == null ? "" : String(defaults.maxSteps);
    }
    if (this.trainingElements.checkpointEveryInput && defaults.checkpointEveryEpisodes != null) {
      this.trainingElements.checkpointEveryInput.value = String(defaults.checkpointEveryEpisodes);
    }
    if (this.trainingElements.checkpointDirInput) {
      this.trainingElements.checkpointDirInput.value = defaults.checkpointDir || "";
    }
    if (this.trainingElements.loadModelPathInput) {
      this.trainingElements.loadModelPathInput.value = defaults.loadModelPath || "";
    }
    if (this.trainingElements.terminateCheckbox) {
      this.trainingElements.terminateCheckbox.checked = !!defaults.terminateOnWin;
    }
    if (this.trainingElements.playOnlyCheckbox) {
      this.trainingElements.playOnlyCheckbox.checked = !!defaults.playOnly;
    }
    this.trainingDefaultsApplied = true;
  }

  var wasRunning = this.isTrainingRunning;
  var running = !!status.running;
  this.isTrainingRunning = running;
  var requested = status.requestedEpisodes || 0;
  var completed = status.completedEpisodes || 0;
  var avg = status.averageScore || 0;
  var maxTile = status.maxTileSeen || 0;
  var latestFrameId = status.latestFrameId || 0;
  var ackedFrameId = status.ackedFrameId || 0;
  var syncWithFrontend = !!status.syncWithFrontend;
  if (!syncWithFrontend && this.trainingAckTimer) {
    window.clearTimeout(this.trainingAckTimer);
    this.trainingAckTimer = null;
  }

  var lines = [
    "algorithm: " + (status.algorithm || "unknown"),
    "network: " + (status.network || "unknown"),
    "encoding: " + (status.encoding || "unknown"),
    "running: " + (running ? "yes" : "no"),
    "currentEpisode: " + (status.currentEpisode || 0),
    "progress: " + completed + "/" + requested,
    "workers: " + (status.workers || 1),
    "averageScore: " + (typeof avg === "number" ? avg.toFixed(2) : avg),
    "maxTileSeen: " + maxTile,
    "entropy: " + (status.entropy == null ? "-" : status.entropy.toFixed(6)),
    "loss: " + (status.loss == null ? "-" : status.loss.toFixed(6)),
    "globalStep: " + (status.globalStep || 0),
    "syncWithFrontend: " + (syncWithFrontend ? "yes" : "no"),
    "frame: " + latestFrameId,
    "ackedFrame: " + ackedFrameId,
    "awaitingAck: " + (status.awaitingAck ? "yes" : "no"),
    "coolingDown: " + (status.coolingDown ? "yes" : "no"),
    "postAckDelaySec: " + (status.postAckDelaySec == null ? "-" : status.postAckDelaySec),
    "tensorboardEnabled: " + (status.tensorboardEnabled ? "yes" : "no"),
    "tensorboardLogDir: " + (status.tensorboardLogDir || "-"),
    "tensorboardRunDir: " + (status.tensorboardRunDir || "-"),
    "playOnly: " + (status.playOnly ? "yes" : "no"),
    "checkpointEveryEpisodes: " + (status.checkpointEveryEpisodes == null ? "-" : status.checkpointEveryEpisodes),
    "checkpointDir: " + (status.checkpointDir || "-"),
    "checkpointsSaved: " + (status.checkpointsSaved == null ? "-" : status.checkpointsSaved),
    "latestCheckpointPath: " + (status.latestCheckpointPath || "-"),
    "loadModelPath: " + (status.loadModelPath || "-"),
    "loadedModelPath: " + (status.loadedModelPath || "-")
  ];

  if (status.lastEpisode) {
    lines.push(
      "lastEpisode: #" + status.lastEpisode.episode +
      " score=" + status.lastEpisode.score +
      " maxTile=" + status.lastEpisode.maxTile
    );
  }

  if (status.error) {
    lines.push("error: " + status.error);
  } else if (status.tensorboardWarning) {
    lines.push("tensorboardWarning: " + status.tensorboardWarning);
  } else if (status.stopped && !running) {
    lines.push("status: stopped");
  } else if (!running && completed === requested && requested > 0) {
    lines.push("status: completed");
  }

  this.trainingElements.statusBlock.textContent = lines.join("\n");

  if (status.latestState && status.latestState.rawBoard) {
    this.trainingElements.boardBlock.textContent = this.formatTrainingBoard(status.latestState.rawBoard);
    if (running && latestFrameId > this.lastRenderedFrameId) {
      this.renderTrainingOnMainBoard(status.latestState);
      this.lastRenderedFrameId = latestFrameId;
      if (syncWithFrontend) {
        this.scheduleTrainingStepDone(latestFrameId);
      }
    }
  } else {
    this.trainingElements.boardBlock.textContent = "No training board yet.";
  }

  this.trainingElements.startButton.disabled = running;
  this.trainingElements.stopButton.disabled = !running;

  if (running) {
    this.startTrainingPolling();
  } else {
    this.stopTrainingPolling();
    this.lastTrainingRawBoard = null;
    this.lastRenderedFrameId = 0;
    this.lastAckedFrameId = 0;
    if (wasRunning) {
      this.fetchState();
    }
  }
};

// ===================================================================
// Replay functionality
// ===================================================================

BackendGameClient.prototype.setupReplayUI = function () {
  var self = this;
  this.replayElements = {
    select: document.getElementById("replay-select"),
    refreshButton: document.getElementById("replay-refresh-button"),
    playButton: document.getElementById("replay-play-button"),
    pauseButton: document.getElementById("replay-pause-button"),
    stepButton: document.getElementById("replay-step-button"),
    stopButton: document.getElementById("replay-stop-button"),
    speedSlider: document.getElementById("replay-speed"),
    speedLabel: document.getElementById("replay-speed-label"),
    statusBlock: document.getElementById("replay-status")
  };

  if (!this.replayElements.select) {
    return;
  }

  this.replayActive = false;
  this.replayPlaying = false;
  this.replayTimer = null;
  this.replayBestScore = 0;

  this.replayElements.refreshButton.addEventListener("click", function () {
    self.fetchReplayList();
  });
  this.replayElements.playButton.addEventListener("click", function () {
    if (!self.replayActive) {
      self.loadReplay();
    } else {
      self.startReplayAuto();
    }
  });
  this.replayElements.pauseButton.addEventListener("click", function () {
    self.pauseReplay();
  });
  this.replayElements.stepButton.addEventListener("click", function () {
    self.replayStepOnce();
  });
  this.replayElements.stopButton.addEventListener("click", function () {
    self.stopReplay();
  });
  this.replayElements.speedSlider.addEventListener("input", function () {
    var v = parseInt(self.replayElements.speedSlider.value, 10);
    self.replayElements.speedLabel.textContent = v + "x";
    if (self.replayPlaying) {
      self.pauseReplay();
      self.startReplayAuto();
    }
  });

  this.fetchReplayList();
};

BackendGameClient.prototype.fetchReplayList = function () {
  var self = this;
  this.sendRequest("GET", "/api/replay/list", null, function (error, data) {
    if (error) {
      self.logError(error);
      return;
    }
    var select = self.replayElements.select;
    select.innerHTML = '<option value="">-- Select a replay --</option>';
    var replays = data.replays || [];
    replays.forEach(function (r) {
      var opt = document.createElement("option");
      opt.value = r.filename;
      var label = r.filename.replace(/\.json$/, "") +
        " (score=" + r.score + " tile=" + r.maxTile + " steps=" + r.steps + ")";
      opt.textContent = label;
      select.appendChild(opt);
    });
  });
};

BackendGameClient.prototype.loadReplay = function () {
  var self = this;
  var filename = this.replayElements.select.value;
  if (!filename) {
    return;
  }
  this.sendRequest("POST", "/api/replay/load", { filename: filename }, function (error, status) {
    if (error) {
      self.logError(error);
      return;
    }
    self.replayActive = true;
    self.replayBestScore = 0;
    self.updateReplayButtons(true, false);
    self.renderReplayStatus(status);
    // Render initial board from state if available
    if (status.state) {
      self.renderReplayState(status.state);
    }
    // Auto-start playing
    self.startReplayAuto();
  });
};

BackendGameClient.prototype.startReplayAuto = function () {
  if (this.replayTimer) {
    return;
  }
  this.replayPlaying = true;
  this.updateReplayButtons(true, true);
  var speed = parseInt(this.replayElements.speedSlider.value, 10) || 5;
  var interval = Math.max(20, Math.round(200 / speed));
  var self = this;
  this.replayTimer = window.setInterval(function () {
    self.replayStepOnce();
  }, interval);
};

BackendGameClient.prototype.pauseReplay = function () {
  this.replayPlaying = false;
  if (this.replayTimer) {
    window.clearInterval(this.replayTimer);
    this.replayTimer = null;
  }
  this.updateReplayButtons(true, false);
};

BackendGameClient.prototype.replayStepOnce = function () {
  var self = this;
  this.sendRequest("POST", "/api/replay/step", null, function (error, data) {
    if (error) {
      self.logError(error);
      self.pauseReplay();
      return;
    }
    if (data.error || data.finished) {
      self.pauseReplay();
      if (data.finished) {
        self.replayElements.statusBlock.textContent =
          "Replay finished. Step " + data.currentStep + "/" + data.totalSteps +
          "\nFinal score: " + (data.score || 0) + "  MaxTile: " + (data.maxTile || 0);
      }
      return;
    }
    self.renderReplayStatus(data);
    if (data.state) {
      self.renderReplayState(data.state);
    }
  });
};

BackendGameClient.prototype.stopReplay = function () {
  var self = this;
  this.pauseReplay();
  this.replayActive = false;
  this.sendRequest("POST", "/api/replay/stop", null, function (error) {
    if (error) {
      self.logError(error);
    }
  });
  this.updateReplayButtons(false, false);
  this.replayElements.statusBlock.textContent = "No replay loaded.";
  this.fetchState();
};

BackendGameClient.prototype.renderReplayState = function (state) {
  if (!state) {
    return;
  }
  var score = state.score || 0;
  if (score > this.replayBestScore) {
    this.replayBestScore = score;
  }

  var grid = null;
  if (state.animationGrid) {
    grid = this.buildGridFromAnimationGrid(state.animationGrid);
  }
  if (!grid && state.grid) {
    grid = this.buildGrid(state.grid);
  }
  if (!grid) {
    return;
  }

  this.actuator.continueGame();
  this.actuator.actuate(grid, {
    score: score,
    over: !!state.over,
    won: !!state.won,
    bestScore: this.replayBestScore,
    terminated: false
  });
};

BackendGameClient.prototype.renderReplayStatus = function (data) {
  var dirNames = { 0: "Up", 1: "Right", 2: "Down", 3: "Left" };
  var lines = [
    "file: " + (data.filename || "-"),
    "algorithm: " + (data.algorithm || "-") + "  depth: " + (data.depth || "-"),
    "step: " + (data.currentStep || 0) + "/" + (data.totalSteps || 0),
    "score: " + (data.score || 0) + "  maxTile: " + (data.maxTile || 0)
  ];
  if (data.action !== undefined && data.action !== null) {
    lines.push("action: " + (dirNames[data.action] || data.action));
  }
  this.replayElements.statusBlock.textContent = lines.join("\n");
};

BackendGameClient.prototype.updateReplayButtons = function (loaded, playing) {
  this.replayElements.playButton.disabled = playing;
  this.replayElements.pauseButton.disabled = !playing;
  this.replayElements.stepButton.disabled = !loaded || playing;
  this.replayElements.stopButton.disabled = !loaded;
};

BackendGameClient.prototype.logError = function (error) {
  if (window.console && window.console.error) {
    window.console.error(error);
  }
};
