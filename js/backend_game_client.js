function BackendGameClient(InputManager, Actuator) {
  this.inputManager = new InputManager();
  this.actuator = new Actuator();

  this.inFlight = false;
  this.pendingDirection = null;

  this.inputManager.on("move", this.move.bind(this));
  this.inputManager.on("restart", this.restart.bind(this));
  this.inputManager.on("keepPlaying", this.keepPlaying.bind(this));

  this.fetchState();
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

BackendGameClient.prototype.logError = function (error) {
  if (window.console && window.console.error) {
    window.console.error(error);
  }
};
