# 2048
A small clone of [1024](https://play.google.com/store/apps/details?id=com.veewo.a1024), based on [Saming's 2048](http://saming.fr/p/2048/) (also a clone). 2048 was indirectly inspired by [Threes](https://asherv.com/threes/).

Made just for fun. [Play it here!](http://gabrielecirulli.github.io/2048/)

The official app can also be found on the [Play Store](https://play.google.com/store/apps/details?id=com.gabrielecirulli.app2048) and [App Store!](https://itunes.apple.com/us/app/2048-by-gabriele-cirulli/id868076805)

### Contributions

[Anna Harren](https://github.com/iirelu/) and [sigod](https://github.com/sigod) are maintainers for this repository.

Other notable contributors:

 - [TimPetricola](https://github.com/TimPetricola) added best score storage
 - [chrisprice](https://github.com/chrisprice) added custom code for swipe handling on mobile
 - [marcingajda](https://github.com/marcingajda) made swipes work on Windows Phone
 - [mgarciaisaia](https://github.com/mgarciaisaia) added support for Android 2.3

Many thanks to [rayhaanj](https://github.com/rayhaanj), [Mechazawa](https://github.com/Mechazawa), [grant](https://github.com/grant), [remram44](https://github.com/remram44) and [ghoullier](https://github.com/ghoullier) for the many other good contributions.

### Screenshot

<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/1175750/8614312/280e5dc2-26f1-11e5-9f1f-5891c3ca8b26.png" alt="Screenshot"/>
</p>

That screenshot is fake, by the way. I never reached 2048 :smile:

## Contributing
Changes and improvements are more than welcome! Feel free to fork and open a pull request. Please make your changes in a specific branch and request to pull into `master`! If you can, please make sure the game fully works before sending the PR, as that will help speed up the process.

You can find the same information in the [contributing guide.](https://github.com/gabrielecirulli/2048/blob/master/CONTRIBUTING.md)

## License
2048 is licensed under the [MIT license.](https://github.com/gabrielecirulli/2048/blob/master/LICENSE.txt)

## Run locally (Python backend)
This project now includes a Python server that owns all game state and logic.
The browser only sends input commands and renders the state returned by the server.

From the project root:

```bash
python server.py --host 127.0.0.1 --port 8080
```

Then open:

```text
http://127.0.0.1:8080
```

The web page includes a **Headless Training** panel that talks to backend APIs:
- `POST /api/train/start`
- `POST /api/train/stop`
- `GET /api/train/status`
- `POST /api/train/step-done` (frontend animation ack for strict step sync)

Training implementation details:
- algorithm: REINFORCE (policy gradient)
- state encoder: one-hot tensor `16 x 4 x 4` (`2^0` channel represents empty cells)
- policy network: shallow CNN, `2x2` kernels, stride `1`, **no padding**, **no pooling**
- current backend RL trainer supports `workers=1`
- training runs in strict sync mode: backend waits each step until frontend reports animation finished
- and then enforces an additional fixed 2-second delay before the next step

## Headless mode (pure Python)
You can run game episodes without starting the web server:

```bash
python -m backend.headless --episodes 10 --seed 42
```

Run multiple episodes in parallel:

```bash
python -m backend.headless --episodes 1000 --workers 8 --seed 42
```

Headless defaults:
- `state` / `board`: `log2` encoded 4x4 matrix (empty cell = `0`)
- action space: only `0,1,2,3` (up/right/down/left), regardless of whether a move changes the board
- `terminate_on_win=True` by default (use `--no-terminate-on-win` to continue after 2048)

For programmatic use:

```python
from backend.headless import Headless2048Env

env = Headless2048Env(seed=42, terminate_on_win=True)
state = env.reset()
state, reward, terminated, truncated, info = env.step(0)
```

## Donations
I made this in my spare time, and it's hosted on GitHub (which means I don't have any hosting costs), but if you enjoyed the game and feel like buying me coffee, you can donate at my BTC address: `1Ec6onfsQmoP9kkL3zkpB6c5sA4PVcXU2i`. Thank you very much!
