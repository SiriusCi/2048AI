function BackendGameClient(InputManager, Actuator) {
  this.inputManager = new InputManager();
  this.actuator = new Actuator();

  this.inFlight = false;
  this.pendingDirection = null;
  this.trainingPollTimer = null;
  this.trainingElements = null;

  this.inputManager.on("move", this.move.bind(this));
  this.inputManager.on("restart", this.restart.bind(this));
  this.inputManager.on("keepPlaying", this.keepPlaying.bind(this));

  this.setupTrainingUI();
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

  if (!state.terminated) {
    this.actuator.continueGame();
  }

  this.actuator.actuate(this.buildGrid(state.grid), {
    score: state.score || 0,
    over: !!state.over,
    won: !!state.won,
    bestScore: state.bestScore || 0,
    terminated: !!state.terminated
  });
};

BackendGameClient.prototype.restart = function () {
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
  var terminateCheckbox = document.getElementById("training-terminate-on-win");

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
    terminateCheckbox: terminateCheckbox
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

  var payload = {
    episodes: isNaN(episodes) ? 100 : episodes,
    workers: isNaN(workers) ? 1 : workers,
    seed: seedText === "" ? null : parseInt(seedText, 10),
    maxSteps: maxStepsText === "" ? null : parseInt(maxStepsText, 10),
    terminateOnWin: this.trainingElements.terminateCheckbox.checked
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
  }, 1000);
};

BackendGameClient.prototype.stopTrainingPolling = function () {
  if (this.trainingPollTimer) {
    window.clearInterval(this.trainingPollTimer);
    this.trainingPollTimer = null;
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

BackendGameClient.prototype.updateTrainingStatus = function (status) {
  if (!this.trainingElements || !status) {
    return;
  }

  var running = !!status.running;
  var requested = status.requestedEpisodes || 0;
  var completed = status.completedEpisodes || 0;
  var avg = status.averageScore || 0;
  var maxTile = status.maxTileSeen || 0;

  var lines = [
    "algorithm: " + (status.algorithm || "unknown"),
    "network: " + (status.network || "unknown"),
    "encoding: " + (status.encoding || "unknown"),
    "running: " + (running ? "yes" : "no"),
    "progress: " + completed + "/" + requested,
    "workers: " + (status.workers || 1),
    "averageScore: " + (typeof avg === "number" ? avg.toFixed(2) : avg),
    "maxTileSeen: " + maxTile,
    "entropy: " + (status.entropy == null ? "-" : status.entropy.toFixed(6)),
    "loss: " + (status.loss == null ? "-" : status.loss.toFixed(6)),
    "globalStep: " + (status.globalStep || 0)
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
  } else if (status.stopped && !running) {
    lines.push("status: stopped");
  } else if (!running && completed === requested && requested > 0) {
    lines.push("status: completed");
  }

  this.trainingElements.statusBlock.textContent = lines.join("\n");

  if (status.latestState && status.latestState.rawBoard) {
    this.trainingElements.boardBlock.textContent = this.formatTrainingBoard(status.latestState.rawBoard);
  } else {
    this.trainingElements.boardBlock.textContent = "No training board yet.";
  }

  this.trainingElements.startButton.disabled = running;
  this.trainingElements.stopButton.disabled = !running;

  if (running) {
    this.startTrainingPolling();
  } else {
    this.stopTrainingPolling();
  }
};

BackendGameClient.prototype.logError = function (error) {
  if (window.console && window.console.error) {
    window.console.error(error);
  }
};
